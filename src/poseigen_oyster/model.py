import numpy as np

import torch
import torch.nn as nn

import poseigen_seaside.basics as se
import poseigen_trident.utils as tu
import poseigen_trident.prongs as tp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class  Oyster(nn.Module): 
        
    def __init__(self, 
                 dim_i = (1,249,4), dim_f = (1,1,1),
                 incl_only = None,

                 DS = False, DS_wwp_func = nn.AvgPool2d, DS_act = False,

                 
                 kE_k = 15, kE_cf_m = 100, kE_cf_ns = 1, 
                 kE_ck_base = 4, kE_OneByOne = False, 
                 kE_ck_grouped = False, 

                 P_ck = None, P_center = False,
                 
                 O_mods = None, O_mods_ns = 0.3,
                 O_cf_pu_m = None, O_cf_ns = 1, 
                 O_ck_base = None, O_pool_k2s = 0.5,
                 O_pool_func = None, O_actb4pool = True, 
                 O_dropout = 0, O_bias = False,
                           
                 activations = nn.ReLU(), activation_f = None, 
                 batchnorm = 'before'):
        
        super(Oyster, self).__init__()


        # 25.01.25 OCU Version A
        # "Head" == now "kE" for k-mer Embedding
        # "C" == now "P" for position
        # "D" == now "O" for output
        #kE_cf_m == the filter multiplier. 
        #P_center == whether to add padding to center things or not. 

        #25.03.28: added incl_only which uses only a subset of the data. 


        self.incl_only = None
        if incl_only is not None: 
            if len(incl_only) == dim_i[0]: pass
            else: 
                dim_i = (len(incl_only), dim_i[1], dim_i[2])
                self.incl_only = incl_only

                print(dim_i)
            
        


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

        ################################################################

        anti_ref_layers = [tu.FlipLayer(), tu.WWPLayer(func = DS_wwp_func)]
        if DS_act: anti_ref_layers.append(activations)

        self.AntiReflect = nn.Sequential(*anti_ref_layers) if DS else nn.Identity()

        ###########################################################################################

        P_func = nn.AvgPool2d

        if P_ck == None: P_ck = 0
        if P_ck == 0:
            # Now this means do not pool. 
            self.P, P_length = (nn.Identity(), dim_kE_length)
        
        elif P_ck > 1: 

            needed = P_ck - (dim_kE_length % P_ck)
            pad2add = (needed // 2) if P_center else 0

            self.P = P_func((P_ck, 1), stride = (P_ck,1), 
                            padding = (pad2add, 0), ceil_mode = True, divisor_override = P_ck)
            
            P_length = int(np.ceil(dim_kE_length / P_ck))
        
        else: # means its 1, means we global pool

            self.P = P_func((dim_kE_length, 1), stride = (1,1), 
                            padding = (0, 0), ceil_mode = True, divisor_override = dim_kE_length)

            P_length = 1

        if O_cf_pu_m == None: O_cf_pu_m = 0
        O_cf_pu = O_cf_pu_m * kE_cf_f if O_cf_pu_m != 0 else None

        O_Prong_args = {'dim_i': (kE_cf_f, P_length, 1), 'dim_f': dim_f,
                        'mods': O_mods, 'mods_ns': O_mods_ns,    
                        'cf_i': None, 'cf_pu': O_cf_pu,
                        'cf_ns': O_cf_ns, 'ck_base': O_ck_base, 
                        'doublestranded': False, 
                        'ck_dynamic': False, 'skip_first_ck': False,
                        'pool_k2s': O_pool_k2s, 'pool_func': O_pool_func, 'actb4pool': O_actb4pool,
                        'activations': nn.ReLU(), 'activation_f': None,
                        'batchnorm': batchnorm, 'dropout': O_dropout, 'bias': O_bias,
                        'out': True}
    
        self.O = nn.Sequential(*tp.Prong_Y(**O_Prong_args))

        self.actf = nn.Identity() if activation_f is None else activation_f
        
    def forward(self, x):

        if self.incl_only is not None: x = x[:, self.incl_only]

        x = self.Reflect(x)
        x = self.kE(x)
        x = self.AntiReflect(x)
        x = self.P(x)
        x = self.O(x)
        x = self.actf(x)
        
        return x
    


    oys_args = {'dim_i': (1,249,4), 'dim_f': (1,1,1),
                'DS': False, 'DS_wwp_func': nn.AvgPool2d, 'DS_act': False,
                'incl_only': None,

                'kE_k': 15, 'kE_cf_m': 100, 'kE_cf_ns': 1, 
                'kE_ck_base': 4, 'kE_OneByOne': False, 
                'kE_ck_grouped': False, 
                
                'P_ck': None, 'P_center': False,
                
                'O_mods': None, 'O_mods_ns': 0.3,
                'O_cf_pu_m': None, 'O_cf_ns': 1, 
                'O_ck_base': None, 'O_pool_k2s': 0.5,
                'O_pool_func': None, 'O_actb4pool': True, 'O_dropout': 0,
                
                'activations': nn.ReLU(), 'activation_f': None, 
                'batchnorm': 'before'}
    



class DualOyster(nn.Module): 
    
    def __init__(self, 
                 dim_f = (1,1,1),

                 A_dim_i = (1,249,4), A_DS = False, A_incl_only = None,
                 A_kE_k = 15, A_kE_cf_m = 100, A_kE_cf_ns = 1, 
                 A_kE_ck_base = 4, A_kE_OneByOne = False, 
                 A_kE_ck_grouped = False, 
                 A_P_ck = None,

                 use_B = True,

                 B_dim_i = (1,249,4), B_DS = False, B_incl_only = None,
                 B_kE_k = 15, B_kE_cf_m = 100, B_kE_cf_ns = 1, 
                 B_kE_ck_base = 4, B_kE_OneByOne = False, 
                 B_kE_ck_grouped = False, 
                 B_P_ck = None,

                DS_wwp_func = nn.AvgPool2d, DS_act = False,
                P_center = False,
                O_mods = None, O_mods_ns = 0.3,
                O_cf_pu_m = None, O_cf_ns = 1, 
                O_ck_base = None, O_pool_k2s = 0.5,
                O_pool_func = None, O_actb4pool = True, O_dropout = 0,
                 activations = nn.ReLU(), activation_f = None, 
                 batchnorm = 'before'):
        
        super(DualOyster, self).__init__()

        oys_shared_args = {'dim_f': dim_f,
                           'DS_wwp_func': DS_wwp_func, 'DS_act': DS_act,
                           'P_center': P_center,
                           'O_mods': O_mods, 'O_mods_ns': O_mods_ns,
                           'O_cf_pu_m': O_cf_pu_m, 'O_cf_ns': O_cf_ns, 
                           'O_ck_base': O_ck_base, 'O_pool_k2s': O_pool_k2s,
                           'O_pool_func': O_pool_func, 'O_actb4pool': O_actb4pool, 'O_dropout': O_dropout,
                           'activations': activations,'batchnorm': batchnorm}
        

        oys_A_args = {'dim_i': A_dim_i, 'DS': A_DS, 'incl_only': A_incl_only,
                   'kE_k': A_kE_k, 'kE_cf_m': A_kE_cf_m, 'kE_cf_ns': A_kE_cf_ns, 
                   'kE_ck_base': A_kE_ck_base, 'kE_OneByOne': A_kE_OneByOne, 
                   'kE_ck_grouped': A_kE_ck_grouped, 
                   'P_ck': A_P_ck,
                   **oys_shared_args}
        
        self.OysterA = Oyster(**oys_A_args)

        self.use_B = use_B 
        if B_incl_only is not None: 
            if len(B_incl_only) == 0: 
                self.use_B = False 
                B_incl_only = None
        if self.use_B == False: print('NOT USING!!!!')  

        

        oys_B_args = {'dim_i': B_dim_i, 'DS': B_DS, 'incl_only': B_incl_only,
                   'kE_k': B_kE_k, 'kE_cf_m': B_kE_cf_m, 'kE_cf_ns': B_kE_cf_ns, 
                   'kE_ck_base': B_kE_ck_base, 'kE_OneByOne': B_kE_OneByOne, 
                   'kE_ck_grouped': B_kE_ck_grouped, 
                   'P_ck': B_P_ck,
                   **oys_shared_args}
        
        self.OysterB = Oyster(**oys_B_args)

        self.FA = activation_f if activation_f != None else nn.Identity()

    def forward(self, x1, x2): 

        x1 = self.OysterA(x1)
        x2 = self.OysterB(x2) if self.use_B else 0

        return self.FA(x1 + x2)


    

DualOyster_BaseDict = {'A_dim_i': [[(1,249,4)],'cat'], #Set as a single dim_i
                      'B_dim_i': [[(11,40,1)],'cat'], #Set as a single dim_i
                      'dim_f': [[(2,1,1)],'cat'], #Set as a single dim_f

                      'A_kE_ck_grouped': [[True, False], 'cat'], #NEED TO BE SET
                      'A_DS': [[True, False], 'cat'],
                      'A_kE_k': [[12, 20], 'int'], 
                      'A_kE_cf_m': [[50, 150], 'int'],
                      'A_kE_cf_ns': [[1, 1.1, 1.2], 'cat'],
                      'A_kE_ck_base': [[4, 5, 6, 7, 8, None], 'cat'], 
                      'A_kE_OneByOne': [[True, False], 'cat'],
                      'A_P_ck': [[6, 20], 'int'],
                      
                      'B_kE_ck_grouped': [[True, False], 'cat'], #NEED TO BE SET
                      'B_DS': [[True, False], 'cat'],
                      'B_kE_k': [[12, 20], 'int'], 
                      'B_kE_cf_m': [[50, 150], 'int'],
                      'B_kE_cf_ns': [[1, 1.1, 1.2], 'float'],
                      'B_kE_ck_base': [[4, 5, 6, 7, 8, None], 'cat'], 
                      'B_kE_OneByOne': [[True, False], 'cat'],
                      'B_P_ck': [[6, 20], 'int'],


                      
                      'DS_wwp_func': [[nn.MaxPool2d], 'cat'],
                      'DS_act': [[False], 'cat'],

                      'P_center': [[True, False], 'cat'],

                      'O_mods': [[None], 'cat'],
                      'O_mods_ns': [[0.5], 'cat'],
                      'O_cf_pu_m': [[None], 'cat'],
                      'O_cf_ns': [[1, 1.1, 1.2], 'cat'],
                      'O_ck_base': [[None], 'cat'],
                      'O_pool_k2s': [[0, 0.5], 'cat'],
                      'O_pool_func': [[nn.MaxPool2d], 'cat'],
                      'O_actb4pool': [[True], 'cat'],
                      'O_dropout': [[0, 0.2], 'cat'],


                      'activations': [[nn.ReLU(), nn.LeakyReLU()], 'cat'],
                      'batchnorm': [[None, 'before'], 'cat']}


def Reset_DualOyster(dualoyster):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 


    for oyster in [dualoyster.OysterA, dualoyster.OysterB]: 
        lke, los = len(oyster.kE), len(oyster.O)

        for i in np.arange(lke): 
            if isinstance(oyster.kE[i], nn.Conv2d): 
                oyster.kE[i].reset_parameters()
        
        for i in np.arange(los): 
            if isinstance(oyster.O[i], nn.Conv2d): 
                oyster.O[i].reset_parameters()
        print('done reset mod')

    return dualoyster