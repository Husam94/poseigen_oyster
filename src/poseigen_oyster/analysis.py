
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

                if hasattr(mo, "Reflect") and callable(getattr(mo, "Reflect")):
                    mo.Reflect()
                
                x = mo.kE(x)
                
                if hasattr(mo, "AntiReflect") and callable(getattr(mo, "AntiReflect")):
                    mo.AntiReflect()
                
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
            sigo = signal[:, iclu, :, 0]
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


















#==================================
''' ANALYZING NON OYSTER MODELS '''

from scipy.stats import mode as scipy_mode

def evaluate_motif_matrix(
    motif_mat,
    total_seq_len,
    num_rand,
    n_kmers,
    model_pns,
    sub_model=None,
    weights=None,
    target=0,
    method='ixg',
    device=None,
    every_other=1,
    batch_size=512,
    random_seqs=None,
    signals=None,
    return_signal_attributions=False,
):
    """
    Evaluate a single motif matrix on sequence attributions.
    If `signals` is provided, a zero signal array of the same shape is used
    as the neutral signal background. If `return_signal_attributions` is True,
    signal attributions computed on that zero background are returned.

    Parameters
    ----------
    signals : np.ndarray, optional
        Signal input for each random sequence, shape (num_rand, *signal_shape).
    return_signal_attributions : bool
        If True and signals is provided, also return signal attributions.

    Returns
    -------
    dict
        {
            'positions': np.ndarray,
            'sampled_positions': np.ndarray,
            'mean': np.ndarray,
            'median': np.ndarray,
            'std': np.ndarray,
            'mode': np.ndarray,
            'values': np.ndarray,
            'kmer_weights': np.ndarray,
            'similarities': np.ndarray,
            # plus signal_* fields if return_signal_attributions=True
        }
    """
    motif_len = len(motif_mat)
    num_pos = total_seq_len - motif_len + 1
    positions = np.arange(num_pos)
    sampled_positions = positions[::every_other]

    if sampled_positions[-1] != positions[-1]:
        sampled_positions = np.append(sampled_positions, positions[-1])

    if random_seqs is None:
        random_seqs = chis.generate_random_seqs(total_seq_len, num_rand)
    assert random_seqs.shape == (num_rand, total_seq_len, 4)

    kmers = chis.sample_kmers_from_motif(motif_mat, n_kmers)
    similarities = np.array([
        chis.score_kmer_similarity(k, motif_mat) for k in kmers
    ])
    kmer_weights = similarities / similarities.sum()

    if signals is not None:
        if signals.shape[0] != num_rand:
            raise ValueError(
                f"signals must have first dimension {num_rand}, got {signals.shape[0]}"
            )
        signals_zero = np.zeros_like(signals)
    else:
        signals_zero = None

    per_position_values = []
    per_position_signal_values = [] if return_signal_attributions else None

    for pos in sampled_positions:
        batch_list = []
        for kmer in kmers:
            seqs_with_kmer = chis.embed_kmer_in_seqs(kmer, random_seqs, pos)
            batch_list.append(seqs_with_kmer)
        batch_seqs = np.concatenate(batch_list, axis=0)

        batch_signals = None
        if signals_zero is not None:
            batch_signals = np.repeat(signals_zero, n_kmers, axis=0)

        if method == 'ixg':
            if batch_signals is not None:
                attr_seqs, attr_signals = batched_ensemble_ixg(
                    model_pns=model_pns,
                    sub_model=sub_model,
                    seqs=batch_seqs,
                    signals=batch_signals,
                    target=target,
                    weights=weights,
                    device=device,
                    batch_size=batch_size,
                )
                if not return_signal_attributions:
                    attr_signals = None
            else:
                attr_seqs = batched_ensemble_ixg(
                    model_pns=model_pns,
                    sub_model=sub_model,
                    seqs=batch_seqs,
                    target=target,
                    weights=weights,
                    device=device,
                    batch_size=batch_size,
                )
                attr_signals = None
        else:
            raise ValueError(f"Unknown method: {method}")

        # --- sequence motif score only ---
        attr_seqs = attr_seqs.reshape(n_kmers, num_rand, 1, total_seq_len, 4)
        motif_attr = (
            attr_seqs[:, :, 0, pos:pos + motif_len, :].sum(axis=(2, 3)) / motif_len
        )
        weighted_per_seq = np.average(motif_attr, axis=0, weights=kmer_weights)
        per_position_values.append(weighted_per_seq)

        # --- signal attributions (optional) ---
        if return_signal_attributions and attr_signals is not None:
            attr_signals = attr_signals.reshape(
                n_kmers, num_rand, *attr_signals.shape[1:]
            )
            signal_scores = np.abs(attr_signals).sum(
                axis=tuple(range(2, attr_signals.ndim))
            )
            weighted_signal_per_seq = np.average(
                signal_scores, axis=0, weights=kmer_weights
            )
            per_position_signal_values.append(weighted_signal_per_seq)

        if pos % 100 == 0:
            print(f"  finished position {pos}")

    per_position_values = np.stack(per_position_values, axis=0)

    mean_sampled = per_position_values.mean(axis=1)
    median_sampled = np.median(per_position_values, axis=1)
    std_sampled = per_position_values.std(axis=1)
    mode_sampled, _ = scipy_mode(
        per_position_values, axis=1, keepdims=False
    )

    def _interp(y):
        return np.interp(
            positions.astype(float),
            sampled_positions.astype(float),
            y,
        )

    result = {
        'positions': positions,
        'sampled_positions': sampled_positions,
        'mean': _interp(mean_sampled),
        'median': _interp(median_sampled),
        'std': _interp(std_sampled),
        'mode': _interp(mode_sampled.astype(float)),
        'values': per_position_values,
        'kmer_weights': kmer_weights,
        'similarities': similarities,
    }

    if return_signal_attributions and per_position_signal_values:
        per_position_signal_values = np.stack(per_position_signal_values, axis=0)

        mean_sig_sampled = per_position_signal_values.mean(axis=1)
        median_sig_sampled = np.median(per_position_signal_values, axis=1)
        std_sig_sampled = per_position_signal_values.std(axis=1)
        mode_sig_sampled, _ = scipy_mode(
            per_position_signal_values, axis=1, keepdims=False
        )

        result.update({
            'signal_mean': _interp(mean_sig_sampled),
            'signal_median': _interp(median_sig_sampled),
            'signal_std': _interp(std_sig_sampled),
            'signal_mode': _interp(mode_sig_sampled.astype(float)),
            'signal_values': per_position_signal_values,
        })

    return result

def evaluate_motif_at_position(
    motif_mat,
    position,
    total_seq_len,
    num_rand,
    n_kmers,
    model_pns,
    sub_model=None,
    weights=None,
    target=0,
    method='ixg',
    device=None,
    batch_size=512,
    random_seqs=None,
    signals=None,
    return_signal_attributions=False,
    seed=None,
):
    """
    Evaluate a single motif matrix at a single position on sequence attributions.
    If `signals` is provided, zero signals are used as neutral background.
    """
    motif_len = len(motif_mat)

    if random_seqs is None:
        random_seqs = chis.generate_random_seqs(
            total_seq_len=total_seq_len,
            num_rand=num_rand,
            seed=seed,
        )
    assert random_seqs.shape == (num_rand, total_seq_len, 4)

    kmers = chis.sample_kmers_from_motif(motif_mat, n_kmers)
    similarities = np.array([
        chis.score_kmer_similarity(k, motif_mat) for k in kmers
    ])
    kmer_weights = similarities / similarities.sum()

    batch_list = []
    for kmer in kmers:
        seqs_with_kmer = chis.embed_kmer_in_seqs(kmer, random_seqs, position)
        batch_list.append(seqs_with_kmer)
    batch_seqs = np.concatenate(batch_list, axis=0)

    if signals is not None:
        if signals.shape[0] != num_rand:
            raise ValueError(
                f"signals must have first dimension {num_rand}, got {signals.shape[0]}"
            )
        batch_signals = np.repeat(np.zeros_like(signals), n_kmers, axis=0)
    else:
        batch_signals = None

    if method == 'ixg':
        if batch_signals is not None:
            attr_seqs, attr_signals = batched_ensemble_ixg(
                model_pns=model_pns,
                sub_model=sub_model,
                seqs=batch_seqs,
                signals=batch_signals,
                target=target,
                weights=weights,
                device=device,
                batch_size=batch_size,
            )
            if not return_signal_attributions:
                attr_signals = None
        else:
            attr_seqs = batched_ensemble_ixg(
                model_pns=model_pns,
                sub_model=sub_model,
                seqs=batch_seqs,
                target=target,
                weights=weights,
                device=device,
                batch_size=batch_size,
            )
            attr_signals = None
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- sequence motif score only ---
    attr_seqs = attr_seqs.reshape(n_kmers, num_rand, 1, total_seq_len, 4)
    motif_attr = (
        attr_seqs[:, :, 0, position:position + motif_len, :].sum(axis=(2, 3)) / motif_len
    )
    weighted_per_seq = np.average(motif_attr, axis=0, weights=kmer_weights)

    result = {
        'per_seq_values': weighted_per_seq,
        'mean': float(np.mean(weighted_per_seq)),
        'median': float(np.median(weighted_per_seq)),
        'std': float(np.std(weighted_per_seq)),
        'abs_mean': float(np.mean(np.abs(weighted_per_seq))),
        'position': position,
        'kmer_weights': kmer_weights,
        'similarities': similarities,
    }

    if return_signal_attributions and attr_signals is not None:
        attr_signals = attr_signals.reshape(n_kmers, num_rand, *attr_signals.shape[1:])
        signal_scores = np.abs(attr_signals).sum(
            axis=tuple(range(2, attr_signals.ndim))
        )
        weighted_signal_per_seq = np.average(
            signal_scores, axis=0, weights=kmer_weights
        )

        result.update({
            'signal_per_seq_values': weighted_signal_per_seq,
            'signal_attr_mean': float(np.mean(weighted_signal_per_seq)),
            'signal_attr_median': float(np.median(weighted_signal_per_seq)),
            'signal_attr_std': float(np.std(weighted_signal_per_seq)),
            'signal_attr_abs_mean': float(np.mean(np.abs(weighted_signal_per_seq))),
        })

    return result


def combine_motif_pair(result_fwd, result_rc, std_method='sem'):
    """
    Combine forward and reverse-complement motif results.

    Parameters
    ----------
    result_fwd, result_rc : dict
        Output dictionaries from evaluate_motif_matrix.
    std_method : {'sem', 'rss'}
        'sem'  : sqrt((std_fwd^2 + std_rc^2) / 2)
        'rss'  : sqrt(std_fwd^2 + std_rc^2)

    Returns
    -------
    dict with combined mean, std, median, mode, and positions.
    """
    mean = (result_fwd['mean'] + result_rc['mean']) / 2.0
    median = (result_fwd['median'] + result_rc['median']) / 2.0
    mode = (result_fwd['mode'] + result_rc['mode']) / 2.0

    std_fwd = result_fwd['std']
    std_rc = result_rc['std']

    if std_method == 'sem':
        std = np.sqrt((std_fwd**2 + std_rc**2) / 2.0)
    elif std_method == 'rss':
        std = np.sqrt(std_fwd**2 + std_rc**2)
    else:
        raise ValueError("std_method must be 'sem' or 'rss'")

    return {
        'positions': result_fwd['positions'],
        'mean': mean,
        'median': median,
        'mode': mode,
        'std': std,
        'mean_fwd': result_fwd['mean'],
        'mean_rc': result_rc['mean'],
        'std_fwd': std_fwd,
        'std_rc': std_rc,
    }


#---------- IXG ------------------

from captum.attr import InputXGradient


def ixg_attribution(model_pn, seqs, signals=None, target=None, device=None, sub_model=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = tu.LoadTorch(model_pn)
    if sub_model is not None:
        model = getattr(model, sub_model)
    model.to(device)
    model.eval()

    if isinstance(seqs, np.ndarray):
        seqs = torch.tensor(seqs, dtype=torch.float32)
    if seqs.ndim == 3:
        seqs = seqs.unsqueeze(1)
    seqs = seqs.to(device)

    inputs = seqs
    if signals is not None:
        if isinstance(signals, np.ndarray):
            signals = torch.tensor(signals, dtype=torch.float32)
        signals = signals.to(device)
        inputs = (seqs, signals)

    ixg = InputXGradient(model)
    attr = ixg.attribute(inputs, target=target)

    if signals is not None:
        return tuple(a.detach().cpu().numpy() for a in attr)
    return attr.detach().cpu().numpy()


def ensemble_ixg(model_pns, seqs, signals=None, sub_model=None, target=None,
                 weights=None, device=None):
    attrs = [
        ixg_attribution(pn, seqs, signals=signals, target=target,
                        device=device, sub_model=sub_model)
        for pn in model_pns
    ]

    if signals is not None:
        seq_attrs = np.stack([a[0] for a in attrs], axis=0)
        sig_attrs = np.stack([a[1] for a in attrs], axis=0)

        if weights is not None:
            weights = np.asarray(weights) / np.sum(weights)
            seq_attrs = np.average(seq_attrs, axis=0, weights=weights)
            sig_attrs = np.average(sig_attrs, axis=0, weights=weights)
        else:
            seq_attrs = seq_attrs.mean(axis=0)
            sig_attrs = sig_attrs.mean(axis=0)

        return seq_attrs, sig_attrs

    attrs = np.stack(attrs, axis=0)
    if weights is not None:
        weights = np.asarray(weights) / np.sum(weights)
        attrs = np.average(attrs, axis=0, weights=weights)
    else:
        attrs = attrs.mean(axis=0)
    return attrs


def batched_ensemble_ixg(model_pns, seqs, signals=None, target=None, weights=None,
                         device=None, batch_size=512, sub_model=None):
    seqs = np.asarray(seqs)
    n = seqs.shape[0]
    seq_attrs = []
    sig_attrs = [] if signals is not None else None

    for i in range(0, n, batch_size):
        batch_seqs = seqs[i:i + batch_size]
        batch_signals = signals[i:i + batch_size] if signals is not None else None
        attr = ensemble_ixg(
            model_pns=model_pns,
            seqs=batch_seqs,
            signals=batch_signals,
            sub_model=sub_model,
            target=target,
            weights=weights,
            device=device,
        )
        if signals is not None:
            seq_attrs.append(attr[0])
            sig_attrs.append(attr[1])
        else:
            seq_attrs.append(attr)

    seq_attrs = np.concatenate(seq_attrs, axis=0)
    if signals is not None:
        return seq_attrs, np.concatenate(sig_attrs, axis=0)
    return seq_attrs

