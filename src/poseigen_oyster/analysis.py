
# These functions are specific to EXACT OYSTER

import numpy as np

import poseigen_chisel as chis
import poseigen_trident.utils as tu

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.cluster import KMeans, AgglomerativeClustering
from kneed import KneeLocator


def SigContrib(model, inp, 
                   sub_model = None, 
                   batchsize = 256):

    shapo = inp.shape #the 2nd dimension is always the number of sigs. 
    num_sigs = shapo[1]

    if isinstance(model, list) is False: model = [model]
    #if sub_model is None: sub_model = ''

    lk = len(inp)

    fullbatches = lk // batchsize
    rem = lk % batchsize

    fins = []

    for fb in np.arange(fullbatches + (rem > 0)):

        fo = fb*batchsize
        batch = inp[fo:fo + batchsize]

        fin = []

        for mo in model: 

            if isinstance(mo, str): mo = tu.LoadTorch(mo)

            if sub_model is not None: mo = getattr(mo, sub_model)
            
            with torch.no_grad(): 
                mo.eval()

                ger = 0 if isinstance(mo.O[0], nn.Conv2d) else 1

                Dweight = mo.O[ger].weight
                Dweight = torch.unsqueeze(Dweight, 0)

                x = torch.FloatTensor(batch).to(device)

                x = mo.Reflect(x)
                x = mo.kE(x)
                x = mo.AntiReflect(x)
                x = mo.P(x)
                
                x = torch.unsqueeze(x, axis = 1)
                x = x * Dweight
                x = torch.squeeze(torch.sum(x, axis = 3))

                xshapo = x.shape
                x = torch.sum(x.reshape(xshapo[0], num_sigs, xshapo[-1] // num_sigs), -1)

                fin.append(x.cpu().detach().numpy())

        fin = np.mean(np.stack(fin), axis = 0)
        fins.append(fin) 
        
    return np.vstack(fins)



def kmer_contrib(model, inp, sub_model = None, 
                 joint = False, chunk = False): 
    
    # The inp is (kmer, *kmer shape)

    mo = tu.LoadTorch(model) if isinstance(model, str) else model
    
    if sub_model is not None: mo = getattr(mo, sub_model)

    shapo = inp.shape

    with torch.no_grad(): 
        mo.eval()

        ger = 0 if isinstance(mo.O[0], nn.Conv2d) else 1

        Dweight = mo.O[ger].weight
        Dweight = torch.unsqueeze(Dweight, 0)

        x = inp

        if joint: x = mo.Reflect(x)

        x = mo.kE(x)
        x = torch.unsqueeze(x, axis = 1)

        #---------------------------

        if isinstance(mo.P, nn.Identity) is False:
            P_ck = mo.P.kernel_size[0]
            x = x / P_ck

        #---------------------------

        x = x * Dweight

        if chunk: 
            xshapo = x.shape
            x = x.reshape(xshapo[0], xshapo[1], shapo[1], xshapo[2] // shapo[1], *xshapo[-2:])
            x = torch.sum(x, axis = 3)
        else: 
            x = torch.sum(x, axis = -3)

        if joint: x = torch.unsqueeze(torch.mean(x, axis = -1), dim= -1)

    return x.cpu().detach().numpy()


def kmer_contrib_multi(model, inp, sub_model = None, 
                       joint = False, chunk = False, batchsize = 256):

    lk = len(inp)
    fullbatches = lk // batchsize
    rem = lk % batchsize

    kc_args = {'sub_model': sub_model, 'joint': joint, 'chunk': chunk}

    fins = []

    for fb in np.arange(fullbatches + (rem > 0)):

        fo = fb*batchsize

        batch = inp[fo:fo + batchsize]

        batch_t = torch.FloatTensor(batch).to(device)

        if isinstance(model, list): 
            fin = [kmer_contrib(m, batch_t, ** kc_args) for m in model]
            fin = np.mean(np.stack(fin), axis = 0)
        else: 
            fin = kmer_contrib(model, batch_t, ** kc_args)
        
        fins.append(fin)
    
    fins = np.vstack(fins)

    return fins



def kppm_contrib(model, kppm,
                 
                 sub_model = None, joint = False, chunk = False,
                 score_mode = 'recipeuc',
                 
                 random_seqs = 10000, unique = True, 
                 scoreweight = True, batchsize = 256):

    # generates kmers from the ppm and scores them to the final output
    # ppm must be the same size as the model head 

    kmers = chis.kmersFromPPM(kppm, num = random_seqs, unique = unique)
    kmers_OHE = np.expand_dims(chis.Seqs2OHE(kmers), axis = 1)

    fins = kmer_contrib_multi(model, kmers_OHE, sub_model = sub_model,
                              joint = joint, chunk = chunk, batchsize = batchsize)

    if scoreweight: 

        kmers_scored = chis.PWMScorer(np.squeeze(kmers_OHE, axis = 1), 
                                      kppm, revcomp = joint,
                                      score_mode = score_mode).reshape(-1, 1, 1, 1)
                
        out = np.sum((fins * kmers_scored), axis = 0) / np.sum(kmers_scored)
        
    else: out = np.mean(fins, axis = 0)

    return out

def contrib_perpos(contribs, oys_dict, submodel = None):
   
    # this function takes your contribution per position block 
    # and expands it to the whole sequence

    # center in this case means that the model used padding for pooling. 

    #contribs shape: (subjects, pos, *)

    #-------------------------------

    submodel = '' if submodel is None else submodel + '_'

    seq_len = oys_dict[submodel + 'dim_i'][1]
    k_len, P_ck = [oys_dict[submodel + xoo] for xoo in ['kE_k', 'P_ck']]
    P_center = oys_dict['P_center']

    #-------------------------------

    dim_kE_length = seq_len - k_len + 1

    needed = P_ck - (dim_kE_length % P_ck)
    pad2add = (needed // 2) if P_center else 0

    P_length = int(np.ceil(dim_kE_length / P_ck))

    #-------------------------------------------

    firstpos = P_ck - pad2add
    midpos = P_ck 
    lastpos = dim_kE_length - firstpos - (midpos * (P_length - 2))

    contribs_perpos = [np.repeat(contribs[:, [0]], firstpos, axis = 1),
                       *[np.repeat(contribs[:, x], midpos, axis = 1) for x in [np.arange(P_length)[1:-1]]], 
                       np.repeat(contribs[:, [-1]], lastpos, axis = 1)]
        
    contribs_perpos = np.concatenate(contribs_perpos, axis = 1)
                    
    return contribs_perpos


def sppm_contrib(kppm_contribs_exp, kppm_idxs):

    #This returns a list since they can be different sizes. 

    sizo = kppm_contribs_exp.shape[1]

    outs = []
    for idxs in kppm_idxs: 

        lex = len(idxs)
        sex = sizo-lex

        contrib_cuts = np.array([kppm_contribs_exp[do, ido: sex+ido] 
                                 for ido, do in enumerate(idxs)])

        outs.append(np.sum(contrib_cuts, axis = 0))
    
    return outs

def sppm_avgrevcomp(sppm_contribs): 

    # This averages the revcomps, flips the revcomp first tho 
    # sppm_contribs are of shape (contribs, len) 
    lex = len(sppm_contribs)
    idxs = np.arange(lex)
    sppm1 = [sppm_contribs[ido] for ido in idxs[:lex//2]]
    sppm2 = [sppm_contribs[ido][::-1] for ido in idxs[lex//2:]]

    sppm_avg = [(s1 + s2) / 2 for s1, s2 in zip(sppm1, sppm2)]

    return sppm_avg

def contrib_fill(contribs, seq_len, 
                 fill_with = np.nan, fill_center = True): 

    # contribs is an array or a list 
    expocons = []
    for cont in contribs: 
        diffx = seq_len - len(cont)

        diffx1 = diffx // 2 if fill_center else 0
        diffx2 = diffx - diffx1

        contexp = np.hstack([np.full(diffx1, fill_with), cont, np.full(diffx2, fill_with)])
                             
        expocons.append(contexp)
    
    return np.array(expocons)



def kmer_cluster(inp, H_size,
                 num_clusters = 10, sample = None,
                 chunk = False, 
                 cluster_mode = [KMeans, {}]):

    sig_win_rs = tu.TridentWindow(inp, H_size)

    num_sigs = sig_win_rs.shape[1]
    
    lk = len(sig_win_rs)
    
    if isinstance(sample, int): 
        rando = np.random.choice(np.arange(lk), sample, replace=False)
    else: rando = np.arange(lk)

    if chunk:  #EACH SIGNAL INDIVIDUAL CLUSTER

        clustod = []
        for i in np.arange(num_sigs):

            sig = sig_win_rs[:, i].reshape(lk, -1)
            sig_samp = sig[rando]
            clust = cluster_mode[0](n_clusters = num_clusters, **cluster_mode[1])
            clust.fit(sig_samp)
            clustod.append(clust.predict(sig))
            print(f'finished: {i}')
        
        clustod = np.stack(clustod, axis = 1)
    
    else: 
        clust = cluster_mode[0](n_clusters = num_clusters, **cluster_mode[1])
        clust.fit(sig_win_rs[rando].reshape(len(rando), -1))
        clustod = clust.predict(sig_win_rs.reshape(lk, -1))
    
    return clustod


def cluster_centroids(signal, clusters):
    #signal is a (n, *) shape and clusters is a 1dim
    return np.stack([np.mean(signal[clusters == b], axis = 0) for b in np.unique(clusters)])


def cluster_refine(signal, clusters, S = 1, return_newcents = False):
                   
    #uses agglomerative clsutering on centroids 
    # S is sensitivity for kneed. 

    clust1_uni = np.unique(clusters)

    cents1 = cluster_centroids(signal, clusters)
    model1 = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model1 = model1.fit(cents1.reshape(len(cents1), -1))

    kn = KneeLocator(np.arange(len(model1.distances_)), model1.distances_, 
                     S = S, 
                     curve='convex', direction='increasing')
    kno = kn.knee

    model2 = AgglomerativeClustering(distance_threshold=model1.distances_[kno], n_clusters=None)
    clust2 = model2.fit_predict(cents1.reshape(len(cents1), -1)) #you have a lst the length of unique clusters


    clust2_uni = np.unique(clust2)
    
    print(f'from {len(clust1_uni)} to {len(clust2_uni)}')

    clust2_sepidx = [clust1_uni[clust2 == u] for u in clust2_uni] #This now tells you which of the original clusters belongs to each new cluster 


    newclust = np.ones(len(clusters))
    for u,g in zip(clust2_uni, clust2_sepidx): 
        for h in g:
            newclust[clusters == h] = u 
    

    newcents = cluster_centroids(signal, newclust)

    return (newclust, newcents) if return_newcents else newclust


def centroid_contrib(wins, contribs, clusters, 
                     weighted = True, score_mode = 'recipeuc', 
                     chunk = False, return_centroids = False):
    #wins shape = (number, signal, k, 1)
    #contribs shape = (number, signal, pos, 1)
    #clusters shape = (number, signal) 

    #this gets the centroids and their score per position 

    num_sigs = contribs.shape[1]

    centroids_all = []
    contrib_cents_all = []

    if chunk:

        for s in np.arange(num_sigs): 

            centroids = []
            contrib_cents = []

            clus = np.unique(clusters[:, s])

            for c in clus: 

                wino = wins[:, s][clusters[:, s] == c] #this is now (num, k, 1) 

                centroid = np.mean(wino, axis = 0) #this is now (k, 1)
                centroids.append(centroid)

                contribo = contribs[:, s][clusters[:, s] == c] #this is now (num, pos, 1) 

                if weighted:

                    centroid_exp = np.expand_dims(centroid, axis = 0)

                    if score_mode == 'dot': 
                        scores = np.sum(wino * centroid_exp, axis = (-1, -2))

                    elif score_mode == 'recipeuc': 
                        scores = 1 / (np.sum(np.abs(wino-centroid_exp)**2, axis = (-1, -2))**(1/2))
                    
                    scores = scores.reshape(-1, 1, 1)
                    
                    contrib_cent = np.sum(contribo * scores, axis = 0) / np.sum(scores)
                
                else: 

                    contrib_cent = np.mean(contribo, axis = 0)
                
                contrib_cents.append(contrib_cent)

            centroids_all.append(np.stack(centroids)) 
            contrib_cents_all.append(np.stack(contrib_cents))
    
    else: 

        for c in np.unique(clusters): 

            wino = wins[clusters == c] #this is now (num, s, k, 1) 

            centroid = np.mean(wino, axis = 0) #this is now (s, k, 1)
            centroids_all.append(centroid)

            contribo = contribs[clusters == c, 0] #this is now (num, pos, 1)

            if weighted: #Compare the wino to the centroid. Closer one is weighted more. 

                centroid_exp = np.expand_dims(centroid, axis = 0)

                if score_mode == 'dot': 
                    scores = np.sum(wino * centroid_exp, axis = (-1, -2, -3))

                elif score_mode == 'recipeuc': 
                    scores = 1 / (np.sum(np.abs(wino-centroid_exp)**2, axis = (-1, -2, -3))**(1/2))
                
                scores = scores.reshape(-1, 1, 1)
                
                contrib_cent = np.sum(contribo * scores, axis = 0) / np.sum(scores)
            
            else: 

                contrib_cent = np.mean(contribo, axis = 0)
            
            contrib_cents_all.append(contrib_cent)
        
        centroids_all, contrib_cents_all = [np.array(x) 
                                            for x in [centroids_all, contrib_cents_all]]

    return (centroids_all, contrib_cents_all) if return_centroids else contrib_cents_all


