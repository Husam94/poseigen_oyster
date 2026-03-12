
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
                   batchsize = 256, model_weights = None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        if model_weights is None: fin = np.mean(np.stack(fin), axis = 0)
        else: fin = np.average(np.stack(fin), axis = 0)
        
        fins.append(fin) 
        
    return np.vstack(fins)



def kmer_contrib(model, inp, sub_model = None, 
                 dim_kE_length = None,
                 joint = False, chunk = False): 
    
    # The inp is (kmer, *kmer shape)

    # Bias term in the dense layer is spread out to kE to simplify interpretation. 
    # Especially useful for dual oyster which can have different bias

    mo = tu.LoadTorch(model) if isinstance(model, str) else model
    
    if sub_model is not None: mo = getattr(mo, sub_model)

    shapo = inp.shape


    with torch.no_grad(): 
        mo.eval()

        #---------------------------------

        ger = 0 if isinstance(mo.O[0], nn.Conv2d) else 1

        Dweight = mo.O[ger].weight
        Dweight = torch.unsqueeze(Dweight, 0)

        Dbias = mo.O[ger].bias
        if Dbias is None: Dbias = 0
        else: 
            if dim_kE_length is None: dim_kE_length = mo.dim_kE_length
            Dbias = Dbias / dim_kE_length
        
        #---------------------------------
        
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
            x = torch.sum(x, axis = -3, keepdim=True)

        if joint: x = torch.unsqueeze(torch.mean(x, axis = -1), dim= -1)
    
        x = x + Dbias

    return x.cpu().detach().numpy()


def kmer_contrib_multi(model, inp, sub_model=None, 
                       joint=False, chunk=False, batchsize=256,
                       model_weights=None, dim_kE_length = None):
    ''' inp can be 4 dimensional (N, *trident) or 5 dimensional (N, win num, *trident) '''

    # Check if inp is 5-dimensional and reshape it to 4-dimensional
    original_shape = inp.shape
    if inp.ndim == 5:
        inp = inp.reshape(-1, *original_shape[2:])  # Flatten the first two axes

    lk = len(inp)
    fullbatches = lk // batchsize
    rem = lk % batchsize

    kc_args = {'sub_model': sub_model, 'joint': joint, 'chunk': chunk, 'dim_kE_length': dim_kE_length}

    # Validate/normalize model weights once
    model_weights_arr = None
    if isinstance(model, list) and (model_weights is not None):
        model_weights_arr = np.asarray(model_weights, dtype=float).reshape(-1)
        if len(model_weights_arr) != len(model):
            raise ValueError(
                f"model_weights must have length {len(model)} (got {len(model_weights_arr)})."
            )
        wsum = np.sum(model_weights_arr)
        if wsum == 0:
            raise ValueError("model_weights must sum to a non-zero value.")
        model_weights_arr = model_weights_arr / wsum

    fins = []

    for fb in np.arange(fullbatches + (rem > 0)):
        fo = fb * batchsize
        batch = inp[fo:fo + batchsize]

        batch_t = torch.FloatTensor(batch).to(device)

        if isinstance(model, list):
            fin = [kmer_contrib(m, batch_t, **kc_args) for m in model]
            fin = np.stack(fin, axis=0)

            if model_weights_arr is None:
                fin = np.mean(fin, axis=0)
            else:
                fin = np.average(fin, axis=0, weights=model_weights_arr)
        else:
            if model_weights is not None:
                raise ValueError("model_weights can only be used when model is a list.")
            fin = kmer_contrib(model, batch_t, **kc_args)
        
        fins.append(fin)

    fins = np.vstack(fins)

    fins = fins[:, 0, :, :, 0]

    # Reshape the output back to 5-dimensional if the input was 5-dimensional
    if len(original_shape) > 4:  # Check if the input was 5-dimensional
        fins = fins.reshape(original_shape[0], original_shape[1], *fins.shape[1:])
    
    return fins


def kppm_contrib(model, kppm,
                 
                 sub_model = None, joint = False, chunk = False,
                 score_mode = 'recipeuc',

                 abs = False,
                 
                 random_seqs = 10000, unique = True, 
                 scoreweight = True, batchsize = 256,
                 model_weights = None, dim_kE_length = None):
    
    # generates kmers from the ppm and scores them to the final output
    # ppm must be the same size as the model head 

    kmers = chis.kmersFromPPM(kppm, num = random_seqs, unique = unique)
    kmers_OHE = np.expand_dims(chis.Seqs2OHE(kmers), axis = 1)

    fins = kmer_contrib_multi(model, kmers_OHE, sub_model = sub_model,
                              joint = joint, chunk = chunk, batchsize = batchsize,
                              model_weights = model_weights, dim_kE_length=dim_kE_length)
    
    if abs: 
        fins = np.abs(fins)
        #print(np.median(fins, axis = 0))

    if scoreweight: 

        kmers_scored = chis.PWMScorer(np.squeeze(kmers_OHE, axis = 1), 
                                      kppm, revcomp = joint,
                                      score_mode = score_mode).reshape(-1, 1, 1, 1)
                
        out = np.sum((fins * kmers_scored), axis = 0) / np.sum(kmers_scored)
        
    else: out = np.mean(fins, axis = 0)

    return out

def contrib_perpos(contribs, oys_dict, submodel = None, k_len = None):
   
    # this function takes your contribution per position block 
    # and expands it to the whole sequence

    # center in this case means that the model used padding for pooling. 

    #contribs shape: (subjects, pos, *)

    #-------------------------------

    submodel = '' if submodel is None else submodel + '_'

    seq_len = oys_dict[submodel + 'dim_i'][1]
    P_ck = oys_dict[submodel + 'P_ck']
    
    
    if 'P_center' in oys_dict.keys(): P_center = oys_dict['P_center']
    else: P_center = False

    if k_len is None: k_len = oys_dict[submodel + 'kE_k']

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


def sppm_contrib(kppm_contribs_exp, kppm_idxs, sum = False):

    #This returns a list since they can be different sizes. 

    sizo = kppm_contribs_exp.shape[1]

    funco = np.sum if sum else np.mean

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



def kmer_cluster(inp,
                 num_clusters = 10, sample = None,
                 chunk = False, 
                 cluster_mode = [KMeans, {}]):
    

    ''' inp is in window format: (N, num_win, *trident)'''

    
    original_shape = inp.shape

    inp = inp.reshape(-1, *original_shape[2:])
    num_sigs = inp.shape[1]
    lk = len(inp)
    
    if isinstance(sample, int): 
        rando = np.random.choice(np.arange(lk), sample, replace=False)
    else: rando = np.arange(lk)

    if chunk:  #EACH SIGNAL INDIVIDUAL CLUSTER

        clustod = []
        for i in np.arange(num_sigs):

            sig = inp[:, i].reshape(lk, -1)
            sig_samp = sig[rando]
            clust = cluster_mode[0](n_clusters = num_clusters, **cluster_mode[1])
            clust.fit(sig_samp)
            clustod.append(clust.predict(sig))
            print(f'finished: {i}')
        
        clustod = np.stack(clustod, axis = 1)           # (N, signals)
    
    else: 
        clust = cluster_mode[0](n_clusters = num_clusters, **cluster_mode[1])
        clust.fit(inp[rando].reshape(len(rando), -1))
        clustod = clust.predict(inp.reshape(lk, -1))    # (N,)

        clustod = np.expand_dims(clustod, axis = 1)     # (N, 1)


    clustod = clustod.reshape(*original_shape[:2], *clustod.shape[1:])
    
    return clustod


def cluster_centroids(signal, clusters):

    '''signal is of shape (N, *whatever), clusters is of shape (N)'''

    #signal is a (n, *) shape and clusters is a 1dim
    return np.stack([np.mean(signal[clusters == b], axis = 0) for b in np.unique(clusters)])

def cluster_refine(signal, clusters, S = 1, return_newcents = False):

    ''' 
    uses agglomerative clsutering on centroids 
    signal input = (N, num_win, *trident), clusters input = (N, num_win, num_signals)
    S is sensitivity for kneed

    output clusters is the same shape of clusters input
      '''

    original_shape = signal.shape

    signal = signal.reshape(-1, *original_shape[2:])

    clusters = clusters.reshape(-1, clusters.shape[-1])

    newclust_all, newcents_all = [], []

    for iclu, clustod in enumerate(clusters.T): 
        
        if clusters.shape[1] > 1: 
            exo = (1, *original_shape[3:])
            sigo = signal[:, iclu].reshape(signal.shape, -1)
        else: 
            exo = original_shape[2:]
            sigo = signal.reshape(signal.shape[0], -1)

        clust1_uni = np.unique(clustod)

        cents1 = cluster_centroids(sigo, clustod)                 # (num_clusters, signal flat) 

        model1 = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model1 = model1.fit(cents1)

        kn = KneeLocator(np.arange(len(model1.distances_)), model1.distances_, 
                        S = S, 
                        curve='convex', direction='increasing')
        kno = kn.knee

        model2 = AgglomerativeClustering(distance_threshold=model1.distances_[kno], n_clusters=None)
        clust2 = model2.fit_predict(cents1) 


        clust2_uni = np.unique(clust2)
        
        print(f'from {len(clust1_uni)} to {len(clust2_uni)}')

        clust2_sepidx = [clust1_uni[clust2 == u] for u in clust2_uni] #This now tells you which of the original clusters belongs to each new cluster 

        newclust = np.ones(len(clustod))
        for u,g in zip(clust2_uni, clust2_sepidx): 
            for h in g:
                newclust[clustod == h] = u 

        newclust_rs = newclust.reshape(*original_shape[:2])
        newclust_all.append(newclust_rs)
    

        newcents = cluster_centroids(sigo, newclust)                # (num_clusters, signal flat)
        newcents_rs = newcents.reshape(-1, *exo)
        newcents_all.append(newcents_rs)

    newclust_all = np.stack(newclust_all, axis = -1)

    return (newclust_all, newcents_all) if return_newcents else newclust_all


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


