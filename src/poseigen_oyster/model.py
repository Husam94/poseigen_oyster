import numpy as np

import poseigen_seaside.basics as se
import poseigen_trident.utils as tu
import poseigen_trident.prongs as tp

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Oyster(nn.Module):
    """Oyster model with optional k-mer embedding chunking via ``kE_chunked``."""
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 incl_only = None, DS = False,
                 
                 kE_k = 15, kE_cf_m = 100, kE_cf_ns = 1, 
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 
                 kE_freeze = False,

                 P_ck = None, P_center = False, kE_chunked = None,
                 
                 O_dropout = 0, 
                           
                 activations = nn.ReLU(), activation_f = None, 
                 batchnorm = 'before'):
        
        super(Oyster, self).__init__()

        num_chunks = None
        if kE_chunked is not None:
            num_chunks = int(kE_chunked)
            if isinstance(kE_chunked, bool) or num_chunks != kE_chunked or num_chunks < 1:
                raise ValueError("kE_chunked must be a positive integer or None.")
            P_center = False

        self.incl_only = None
        if incl_only is not None: 
            if len(incl_only) == dim_i[0]: pass
            else: 
                dim_i = (len(incl_only), dim_i[1], dim_i[2])
                self.incl_only = incl_only

                print(dim_i)
       
        #--------------------------------------

        self.Reflect = tu.ReflectLayer(dims = [-1, -2]) if DS else nn.Identity()
        
        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]: kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = (dim_i[0], dim_i[1], dim_i[2]*2) if DS else dim_i
        dim_kE_width = 2 if DS else 1
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {'dim_i': dim_i_ref, 'dim_f': dim_kE,
                         'mods': 0, 'mods_ns': 0,    
                         'cf_i': None, 'cf_ns': kE_cf_ns, 'ck_base': kE_ck_base,
                         'doublestranded': DS, 'OneByOne': kE_OneByOne,        
                         'ck_grouped': kE_ck_grouped,
                         'activations': activations, 'activation_f': None,
                         'batchnorm': batchnorm, 'dropout': None, 'bias': True,
                         'out': False}
        
        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        # Freeze kE layers if k_freeze is True
        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False

        #--------------------------------------

        anti_ref_layers = [tu.FlipLayer(), tu.WWPLayer(func = nn.AvgPool2d)]

        self.AntiReflect = nn.Sequential(*anti_ref_layers) if DS else nn.Identity()

        #--------------------------------------

        P_func = nn.AvgPool2d

        if P_ck == None: P_ck = 0
        if P_ck == 0:
            # Now this means do not pool. 
            self.P, P_length = (nn.Identity(), dim_kE_length)
        
        elif P_ck > 1: 

            P_length = int(np.ceil(dim_kE_length / P_ck))
            if P_center:
                total_padding = (-dim_kE_length) % P_ck
                pad_left = total_padding // 2
                pad_right = total_padding - pad_left
                self.P = nn.Sequential(
                    nn.ZeroPad2d((0, 0, pad_left, pad_right)),
                    P_func((P_ck, 1), stride=(P_ck, 1),
                           padding=0, ceil_mode=False,
                           divisor_override=P_ck))
            else:
                self.P = P_func((P_ck, 1), stride=(P_ck, 1),
                                padding=0, ceil_mode=True,
                                divisor_override=P_ck)
        
        else: # means its 1, means we global pool

            self.P = P_func((dim_kE_length, 1), stride = (1,1), 
                            padding = (0, 0), ceil_mode = True, divisor_override = dim_kE_length)
            P_length = 1

        self.P_ck = P_ck
        self.DS = DS
        self.kE_k = kE_k
        self.dim_kE_length = dim_kE_length
        self.chunk_boundaries = None
        if num_chunks is not None and P_ck > 0:
            self.chunk_boundaries = calculate_chunk_boundaries_simple(
                dim_i_length=dim_i[1],
                kE_k=kE_k,
                P_ck=P_ck,
                num_chunks=num_chunks,
                dim_kE_length=dim_kE_length)



        O_Prong_args = {'dim_i': (kE_cf_f, P_length, 1), 'dim_f': dim_f, 'mods': 1, 
                        'activations': activations, 'activation_f': None,
                        'batchnorm': batchnorm, 'dropout': O_dropout, 
                        'bias': True, 'out': True}
    
        self.O = nn.Sequential(*tp.Prong_Y(**O_Prong_args))

        self.actf = nn.Identity() if activation_f is None else activation_f
        
    def forward(self, x):

        if self.incl_only is not None: x = x[:, self.incl_only]

        x = self.Reflect(x)
        if self.chunk_boundaries is None:
            x = self.P(self.AntiReflect(self.kE(x)))
        else:
            x = self._forward_chunked(x)
        x = self.O(x)
        x = self.actf(x)
        
        return x

    def _forward_chunked(self, x):
        pooled_chunks = []
        pooled_sum = None

        for chunk_info in self.chunk_boundaries:
            conv_start = chunk_info['input_start']
            conv_end = conv_start + chunk_info['conv_length']
            conv_input_end = conv_end + self.kE_k - 1

            if self.DS:
                forward = self.kE(x[:, :, conv_start:conv_input_end, :])
                reverse_start = self.dim_kE_length - conv_end
                reverse_end = self.dim_kE_length - conv_start + self.kE_k - 1
                reverse = self.kE(x[:, :, reverse_start:reverse_end, :])
                chunk_features = (
                    forward[:, :, :, 0]
                    + torch.flip(reverse[:, :, :, 1], dims=[2])
                ).unsqueeze(-1) / 2
            else:
                chunk_features = self.AntiReflect(
                    self.kE(x[:, :, chunk_info['input_start']:chunk_info['input_end'], :]))

            if self.P_ck == 1:
                chunk_sum = chunk_features.sum(dim=2, keepdim=True)
                pooled_sum = chunk_sum if pooled_sum is None else pooled_sum + chunk_sum
            else:
                pooled_chunks.append(self.P(chunk_features))

        if self.P_ck == 1:
            return pooled_sum / self.dim_kE_length
        return torch.cat(pooled_chunks, dim=2)


#==========================================



def calculate_chunk_boundaries_simple(dim_i_length, kE_k, P_ck, num_chunks, dim_kE_length):
    """
    Precompute chunk boundaries for memory-efficient chunked forward pass.
    Simplified version without P_center support.
    
    Args:
        dim_i_length: Input sequence length
        kE_k: Convolution kernel size
        P_ck: Pooling kernel size
        num_chunks: Number of chunks to split into
        dim_kE_length: Convolution output length (dim_i_length - kE_k + 1)
    
    Returns:
        List of dicts containing boundary info for each chunk, or None if chunking not needed
    """
    if not num_chunks or P_ck <= 0:
        return None
    
    # Calculate number of pooling blocks
    num_pools = int(np.ceil(dim_kE_length / P_ck))
    chunk_pools = max(1, int(np.ceil(num_pools / num_chunks)))
    
    # If single chunk suffices, no need for boundaries
    if chunk_pools >= num_pools:
        return None
    
    # Calculate actual number of chunks needed
    actual_num_chunks = int(np.ceil(num_pools / chunk_pools))
    
    # Precompute chunk boundaries
    chunk_boundaries = []
    for chunk_idx in range(actual_num_chunks):
        # Pooling block indices for this chunk
        pool_start = chunk_idx * chunk_pools
        pool_end = min(pool_start + chunk_pools, num_pools)
        
        # Convert to convolution output positions
        conv_start = pool_start * P_ck
        conv_end = min(pool_end * P_ck, dim_kE_length)
        
        # Convert to input indices
        input_start = conv_start
        input_end = conv_end + (kE_k - 1)
        input_end = min(input_end, dim_i_length)
        
        # Calculate conv output length for this chunk
        chunk_conv_length = conv_end - conv_start
        
        chunk_boundaries.append({
            'input_start': input_start,
            'input_end': input_end,
            'conv_length': chunk_conv_length
        })
    
    return chunk_boundaries



    

Oyster_BaseDict = {'dim_i': (1,249,4), 'dim_f': (1,1,1),
                'DS': False,
                'incl_only': None,

                'kE_k': 15, 'kE_cf_m': 100, 'kE_cf_ns': 1, 
                'kE_ck_base': 4, 'kE_OneByOne': False, 
                'kE_ck_grouped': False, 'kE_freeze': False,
                
                'P_ck': None, 'P_center': False, 'kE_chunked': None,
                
                'O_dropout': 0,
                
                'activations': nn.ReLU(), 'activation_f': None, 
                'batchnorm': 'before'}
    


def Reset_Oyster(oyster):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 
    lke, los = len(oyster.kE), len(oyster.O)

    for i in np.arange(lke): 
        if isinstance(oyster.kE[i], nn.Conv2d): 
            oyster.kE[i].reset_parameters()
    
    for i in np.arange(los): 
        if isinstance(oyster.O[i], nn.Conv2d): 
            oyster.O[i].reset_parameters()
    print('done reset mod')

    return oyster




class Dual_Oyster(nn.Module):
    """Combine two Oyster models with independently configured inputs."""

    def __init__(self,
                 dim_f=(1, 1, 1),
                 A_dim_i=(1, 249, 4), A_DS=False, A_incl_only=None,
                 A_kE_k=15, A_kE_cf_m=100, A_kE_cf_ns=1,
                 A_kE_ck_base=4, A_kE_OneByOne=False,
                 A_kE_ck_grouped=False, A_kE_freeze=False,
                 A_P_ck=None, A_P_center=False, A_kE_chunked=None,
                 use_B=True,
                 B_dim_i=(1, 249, 4), B_DS=False, B_incl_only=None,
                 B_kE_k=15, B_kE_cf_m=100, B_kE_cf_ns=1,
                 B_kE_ck_base=4, B_kE_OneByOne=False,
                 B_kE_ck_grouped=False, B_kE_freeze=False,
                 B_P_ck=None, B_P_center=False, B_kE_chunked=None,
                 O_dropout=0, activations=nn.ReLU(),
                 activation_f=None, batchnorm='before'):
        super().__init__()

        shared_args = {
            'dim_f': dim_f,
            'O_dropout': O_dropout,
            'activations': activations,
            'batchnorm': batchnorm,
        }

        self.OysterA = Oyster(
            dim_i=A_dim_i, incl_only=A_incl_only, DS=A_DS,
            kE_k=A_kE_k, kE_cf_m=A_kE_cf_m, kE_cf_ns=A_kE_cf_ns,
            kE_ck_base=A_kE_ck_base, kE_OneByOne=A_kE_OneByOne,
            kE_ck_grouped=A_kE_ck_grouped, kE_freeze=A_kE_freeze,
            P_ck=A_P_ck, P_center=A_P_center, kE_chunked=A_kE_chunked,
            **shared_args)

        self.use_B = use_B
        if B_incl_only is not None and len(B_incl_only) == 0:
            self.use_B = False
            B_incl_only = None

        self.OysterB = Oyster(
            dim_i=B_dim_i, incl_only=B_incl_only, DS=B_DS,
            kE_k=B_kE_k, kE_cf_m=B_kE_cf_m, kE_cf_ns=B_kE_cf_ns,
            kE_ck_base=B_kE_ck_base, kE_OneByOne=B_kE_OneByOne,
            kE_ck_grouped=B_kE_ck_grouped, kE_freeze=B_kE_freeze,
            P_ck=B_P_ck, P_center=B_P_center, kE_chunked=B_kE_chunked,
            **shared_args)

        self.FA = activation_f if activation_f is not None else nn.Identity()

    def forward(self, x1, x2=None):
        output = self.OysterA(x1)
        if self.use_B:
            if x2 is None:
                raise ValueError("x2 is required when use_B=True.")
            output_b = self.OysterB(x2)
            if output.shape != output_b.shape:
                raise ValueError(
                    f"Oyster outputs must have matching shapes; got "
                    f"{tuple(output.shape)} and {tuple(output_b.shape)}.")
            output = output + output_b
        return self.FA(output)


Dual_Oyster_BaseDict = {
    'A_dim_i': [[(1, 249, 4)], 'cat'],
    'B_dim_i': [[(11, 40, 1)], 'cat'],
    'dim_f': [[(2, 1, 1)], 'cat'],

    'A_DS': [[True, False], 'cat'],
    'A_kE_k': [[12, 20], 'int'],
    'A_kE_cf_m': [[50, 150], 'int'],
    'A_kE_cf_ns': [[1, 1.1, 1.2], 'cat'],
    'A_kE_ck_base': [[4, 5, 6, 7, 8, None], 'cat'],
    'A_kE_OneByOne': [[True, False], 'cat'],
    'A_kE_ck_grouped': [[True, False], 'cat'],
    'A_P_ck': [[6, 20], 'int'],
    'A_P_center': [[True, False], 'cat'],
    'A_kE_chunked': [[None, 4], 'cat'],

    'B_DS': [[True, False], 'cat'],
    'B_kE_k': [[12, 20], 'int'],
    'B_kE_cf_m': [[50, 150], 'int'],
    'B_kE_cf_ns': [[1, 1.1, 1.2], 'float'],
    'B_kE_ck_base': [[4, 5, 6, 7, 8, None], 'cat'],
    'B_kE_OneByOne': [[True, False], 'cat'],
    'B_kE_ck_grouped': [[True, False], 'cat'],
    'B_P_ck': [[6, 20], 'int'],
    'B_P_center': [[True, False], 'cat'],
    'B_kE_chunked': [[None, 4], 'cat'],
    
    'O_dropout': [[0, 0.2], 'cat'],
    'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
    'batchnorm': [[None, 'before'], 'cat'],
}


def Reset_Dual_Oyster(dual_oyster):
    Reset_Oyster(dual_oyster.OysterA)
    Reset_Oyster(dual_oyster.OysterB)
    return dual_oyster




