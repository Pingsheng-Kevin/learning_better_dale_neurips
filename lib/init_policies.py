from cmath import e
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
It is expected that you reimplement 

def extra_repr(self):
    r = super().extra_repr()
    return r 

to print out details of the nn.module when you print the model.


Note: Some of the classes use "WeightInitPolicy" in their name, and some just have "Init",
      for simpicity we will move to just using "Init" in future classes
"""
# ------------ Dense Layer Specific ------------
class DenseNormalInit:
    """ 
    Initialises a Dense layer's weights (W) from a normal dist,
    and sets bias to 0.

    Note this is more a combination of Lecun init (just fan-in)
    and He init.

    References:
        https://arxiv.org/pdf/1502.01852.pdf
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    For eg. use numerator=1 for sigmoid, numerator=2 for relu
    """
    def __init__(self, stddev_numerator=2):
        self.stddev_numerator = stddev_numerator

    def init_weights(self, layer):
        nn.init.normal_(layer.W, mean=0, std=np.sqrt((self.stddev_numerator / layer.n_input)))
        nn.init.zeros_(layer.b)
        
class Dense_RNNInit_ColEI_Spectrum_Init:
    
    def __init__(self, ratio=9, num=2):
        self.ratio = ratio
        self.num = num

    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):
        
        ni = layer.n_input // (self.ratio+1)
        ne = layer.n_input - ni
        
        denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))
        # denom = (n_excitation + n_inhibition)

        sigma_e = np.sqrt(1/denom)
        
        # sigma_e = np.sqrt(1/layer.n_input)
        sigma_i = sigma_e * (ne/ni)
        
        We_np = np.random.exponential(scale=sigma_e, size=(layer.n_output, ne))
        Wi_np = -np.random.exponential(scale=sigma_i, size=(layer.n_output, ni))
        W_np= np.concatenate([We_np, Wi_np], axis=1)
        
        _, S, _ = np.linalg.svd(W_np)
        
        temp = np.zeros((layer.n_output, layer.n_input))
        np.fill_diagonal(temp, S)
        S = temp
        
        W_np = np.random.normal(loc=0.0, scale=np.sqrt(self.num/layer.n_input), size=(layer.n_output, layer.n_input))
        U, _, Vh = np.linalg.svd(W_np)
        
        W = U @ S @ Vh
        
        
        layer.W.data = torch.from_numpy(W).float()
        
        # bias
        nn.init.zeros_(layer.b)
        
# class DenseInit:
#     """ 
#     Initialises a Dense layer's weights (W) from a normal dist,
#     and sets bias to 0.

#     Note this is more a combination of Lecun init (just fan-in)
#     and He init.

#     References:
#         https://arxiv.org/pdf/1502.01852.pdf
#         http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

#     For eg. use numerator=1 for sigmoid, numerator=2 for relu
#     """
#     def __init__(self, stddev_numerator=2):
#         self.stddev_numerator = stddev_numerator

#     def init_weights(self, layer):
#         nn.init.normal_(layer.W, mean=0, std=np.sqrt((self.stddev_numerator / layer.n_input)))
#         nn.init.zeros_(layer.b)

class Dense_RNNInit_ColEI_Spectrum_Vector:
    
    def __init__(self, ratio=9, num=2, ratio_vec=9):
        self.ratio = ratio
        self.num = num
        self.ratio_vec = ratio_vec

    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):
        
        ni = layer.n_input // (self.ratio+1)
        ne = layer.n_input - ni
        
        denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))
        # denom = (n_excitation + n_inhibition)

        sigma_e = np.sqrt(1/denom)
        
        # sigma_e = np.sqrt(1/layer.n_input)
        sigma_i = sigma_e * (ne/ni)
        
        We_np = np.random.exponential(scale=sigma_e, size=(layer.n_output, ne))
        Wi_np = -np.random.exponential(scale=sigma_i, size=(layer.n_output, ni))
        W_np= np.concatenate([We_np, Wi_np], axis=1)
        
        _, S, _ = np.linalg.svd(W_np)
        
        temp = np.zeros((layer.n_output, layer.n_input))
        np.fill_diagonal(temp, S)
        S = temp
        
        ni = layer.n_input // (self.ratio_vec+1)
        ne = layer.n_input - ni
        
        denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))
        # denom = (n_excitation + n_inhibition)

        sigma_e = np.sqrt(1/denom)
        
        # sigma_e = np.sqrt(1/layer.n_input)
        sigma_i = sigma_e * (ne/ni)
        
        We_np = np.random.exponential(scale=sigma_e, size=(layer.n_output, ne))
        Wi_np = -np.random.exponential(scale=sigma_i, size=(layer.n_output, ni))
        W_np= np.concatenate([We_np, Wi_np], axis=1)
        
        U, _, Vh = np.linalg.svd(W_np)
        
        W = U @ S @ Vh
        
        
        layer.W.data = torch.from_numpy(W).float()
        
        # bias
        nn.init.zeros_(layer.b)

        
class EiDenseWeightInit_ICLR:
    """ 
    Initialises an EiDense layer's weights as in original paper 
    (note just the inhib_iid_init=False)

    See https://openreview.net/pdf?id=eU776ZYxEpz
    """
    
    def init_weights(self,layer):
        target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))
        if layer.ni == 1: # for example the output layer
            Wix_np = Wex_np.mean(axis=0,keepdims=True) # not random as only one int
            Wei_np = np.ones(shape = (layer.ne, layer.ni))/layer.ni
        else:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.n_input))
            Wei_np = np.ones(shape = (layer.ne, layer.ni))/layer.ni

        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        nn.init.zeros_(layer.b)

class EiDenseWithShunt_WeightInitPolicy_ICLR(EiDenseWeightInit_ICLR):
    def init_weights(self,layer):
        super().init_weights(layer)
        a_numpy = np.sqrt((2*np.pi-1)/layer.n_input) * np.ones(shape=layer.alpha.shape)
        a = torch.from_numpy(a_numpy)
        alpha_val = torch.log(a)
        layer.alpha.data = alpha_val.float()
        
def calc_ln_mu_sigma(mean, var, ex2=None):
    "Given desired mean and var returns ln mu and sigma"
    mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
    sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
    return mu_ln, sigma_ln

class EiDenseWeightInit_WexMean:
    """
    Initialises inhibitory weights to exactly perform the centering operation of Layer Norm.
    
    Sets Wix as copies of the mean row of Wex, Wei is a random vector squashed to sum to 1.  

    Todo: Look at the difference between log normal
    """
    def __init__(self, numerator=2, wex_distribution="exponential"):
        """
        
        """
        self.numerator = numerator
        self.wex_distribution = wex_distribution

    def init_weights(self, layer):
        # this stems from calculating var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] when
        # we have set Wix to mean row of Wex and Wei as summing to 1.
        print(layer.n_input, layer.ne)
        if layer.ne > 1: target_std_wex = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
        else: target_std_wex = np.sqrt(self.numerator/layer.n_input)
        exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
        
        if self.wex_distribution =="exponential":
            Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))
            Wei_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.ni))
        elif self.wex_distribution =="lognormal":
            mu, sigma = calc_ln_mu_sigma(target_std_wex,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(layer.ne, layer.n_input))
            Wei_np = np.random.lognormal(mu, sigma, size=(layer.ne, layer.ni))
        
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(layer.ni,1))*Wex_np.mean(axis=0,keepdims=True)
        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        nn.init.zeros_(layer.b)

class EiDenseWithShunt_Init(EiDenseWeightInit_WexMean): 
    """
    Initialisation for network with forward equations of:

    Z = (1/c + gamma) * g*\hat(z) +b

    Where:
        c is a constant, that protects from division by a small value
        gamma_k = \sum_j wei_kj * alpha_j \sum_i Wix_ji x_i
        alpha = ln(e^\rho +1)

    Init strategy is to initialise:
        alpha = 1-c/ne E[Wex] E[X], therefore
        rho = ln(e^{(1-c)/ne E[Wex] E[X]} -1)

    Note** alpha is not a parameter anymore, so need to change the forward
    methods!!  

    Assumptions:
        X ~ rectified half normal with variance =1, therefore
            E[x] = 1/sqrt(2*pi)
        E[Wex] is the same as std(Wex) and both are equal to:
            sigma = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
    """
    
    def init_weights(self, layer, c=None):
        super().init_weights(layer)
        if c is None: c_np = (5**0.5-1) /2 # golden ratio 0.618....
        else: c_np = c

        e_wex = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
        e_x   = 1/np.sqrt(2*np.pi)
        rho_np = np.log(np.exp(((1-layer.c)/layer.ne*e_wex*e_x)) -1) # torch softplus is alternative
        
        layer.c.data = torch.from_numpy(c_np).float()
        layer.rho.data = torch.from_numpy(rho_np).float()

class EiDenseWeightInit_WexMean_Groups(EiDenseWeightInit_WexMean):
    """
    To implement: if n_groups !=1, we split Wex and Wix into `n_groups` and uses the mean row
                of the Wex "group" for the corresponding Wix "group". But this is to be decided later
    """
    def __init__(self, n_groups=1):
        """
        n_groups : int
        # we might want a tuple with integers specifying the size of the groups 
        """
        self.n_groups = n_groups
                
# ------------ CNN Specific Initializations ------------

# export
class HeConv2d_WeightInitPolicy():
    """
    Remember BaseWeightInitPolicy is basically just nn.Module
    """
    @staticmethod
    def init_weights(conv2d):
        """
        Args:
            conv2d - an instance of nn.Conv2d

        Note this is more a combination of Lecun init (just fan-in)
        and He init (numerator is 2 dues to relu).

        References:
        https://arxiv.org/pdf/1502.01852.pdf
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        """
        fan_in = np.prod(conv2d.weight.shape[1:]) # we scale weights for each filter's activation
        target_std = np.sqrt((2 / fan_in))

        if conv2d.bias is not None:
            nn.init.zeros_(conv2d.bias)

        nn.init.normal_(conv2d.weight, mean=0, std=target_std)
        
class EiConv_WeightInitPolicy():

    def init_weights(self, layer):
        
        # alpha same as before apart from d is defined differently
        a_numpy = np.sqrt((2*np.pi-1)/ (layer.d)) * np.ones(shape=layer.alpha.shape)
        a = torch.from_numpy(a_numpy)
        alpha_val = torch.log(a)
        layer.alpha.data = alpha_val.float() 
        
        # init E and I filter weights
        # for MLP hidden target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))  

        target_std = np.sqrt(2*np.pi/ (layer.d*(2*np.pi-1)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Wex_np = np.random.exponential(scale=exp_scale, size=(layer.e_conv.weight.shape))
        
        # this shouldn't apply to conv models?
        if layer.i_conv.out_channels == 1:
            Wix_np = Wex_np.mean(axis=0, keepdims=True) # not random as only one int
            Wei_np = np.ones(shape = layer.Wei.shape)/layer.i_conv.out_channels
        
        elif layer.i_conv.out_channels != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Wix_np = np.random.exponential(scale=exp_scale, size=(layer.i_conv.weight.shape)) 
            Wei_np = np.ones(shape = layer.Wei.shape)/layer.i_conv.out_channels
            
        layer.e_conv.weight.data = torch.from_numpy(Wex_np).float() 
        layer.i_conv.weight.data = torch.from_numpy(Wix_np).float() 
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        nn.init.zeros_(layer.b)
        nn.init.ones_(layer.g)


# ------------ RNN Specific Initializations ------------

# ------------- "Standard" RNNCell Parameter Initializations ----------
class W_UniformInit:
    """ Initialization of cell.W for h2h for RNNCell from a uniform dist"""

    def __init__(self, num=2):
        self.num = num

    def init_weights(self, cell):
        numerator = np.sqrt(self.num) * np.sqrt(3)
        bound = numerator / np.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.W, a=-bound, b=bound)

class W_NormalInit:
    """ Initialization of cell.W for h2h for RNNCell from a normal dist"""

    def __init__(self, denom=2):
        self.denom = denom

    def init_weights(self, cell):
        nn.init.normal_(cell.W, mean=0, std=np.sqrt((self.denom / cell.n_hidden)))

class U_NormalInit:
    """ Initialization of cell.U for i2h for RNNCell from a normal dist"""

    def __init__(self, denom=2):
        self.denom = denom

    def init_weights(self, cell):
        nn.init.normal_(cell.U, mean=0, std=np.sqrt((self.denom / cell.n_input)))

class W_IdentityInit:
    """
    Identity matrix init of cell.W for h2h for RNNCell:

    see A Simple Way to Initialize Recurrent Networks of Rectified Linear Units
    https://arxiv.org/abs/1504.00941
    """

    @staticmethod
    def init_weights(cell):
        nn.init.eye_(cell.W)
class W_HybridInit_Uniform:
    """
    W = (1-p)*Id + p*W(Uniform Init)
    """
    def __init__(self, p=0.25):
        self.p = p

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        W_p_np = np.random.uniform(-bound, bound, size=(cell.n_hidden, cell.n_hidden))
        W_id_np = np.eye(cell.n_hidden)
        W = self.p * W_p_np + (1-self.p) * W_id_np
        cell.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')
        
class W_OrthogonalInit:
    
    def init_weights(self, cell):
        A = np.random.rand(cell.n_hidden, cell.n_hidden)
        # Perform QR decomposition of the matrix
        Q, R = np.linalg.qr(A)
        cell.W.data = torch.from_numpy(Q).float().to('cuda' if torch.cuda.is_available else 'cpu')
    

class W_ColEIInit_RNN_Spectrum_Init:
    
    def __init__(self, ratio=9, spectral_radius=1.5):
        self.ratio = 9
        self.spectral_radius = spectral_radius
    
    def init_weights(self, layer):
        """
        todo
        """
        ni = layer.n_hidden // (self.ratio + 1)
        ne = layer.n_hidden - ni

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
#         if self.zero_diag: np.fill_diagonal(W, 0)
#         if self.ablate_ii: W[-ni:,-ni:] = 0
        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        # Perform QR decomposition of the matrix
        U, _, Vh = np.linalg.svd(W)

        # generate pytorch init
        bound = 1 / math.sqrt(layer.n_hidden)
        W = np.random.uniform(-bound, bound, size=(layer.n_hidden,layer.n_hidden))
        _, S, _ = np.linalg.svd(W)

        W = U @ np.diag(S) @ Vh
        layer.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')
    
class W_RNNInit_ColEI_Spectrum_Init:
    
    def __init__(self, ratio=9, spectral_radius=1.5):
        self.ratio = 9
        self.spectral_radius = spectral_radius
    
    def init_weights(self, layer):
        """
        todo
        """
        # Pytorch init
        bound = 1 / math.sqrt(layer.n_hidden)
        W = np.random.uniform(-bound, bound, size=(layer.n_hidden,layer.n_hidden))
        U, _, Vh = np.linalg.svd(W)
    
        ni = layer.n_hidden // (self.ratio + 1)
        ne = layer.n_hidden - ni

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
#         if self.zero_diag: np.fill_diagonal(W, 0)
#         if self.ablate_ii: W[-ni:,-ni:] = 0
        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        # Perform QR decomposition of the matrix
        _, S, _ = np.linalg.svd(W)

        W = U @ np.diag(S) @ Vh
        layer.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')
        
class W_RNNInit_ColEI_Spectrum_Vectors:
    
    def __init__(self, ratio=9, spectral_radius=1.5, ratio_vec=9):
        self.ratio = ratio
        self.spectral_radius = spectral_radius
        self.ratio_vec = ratio_vec
    
    def init_weights(self, layer):
        """
        todo
        """
        # Pytorch init
        ni = layer.n_hidden // (self.ratio_vec + 1)
        ne = layer.n_hidden - ni

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)

        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)
        #------
        U, _, Vh = np.linalg.svd(W)
        #------
        ni = layer.n_hidden // (self.ratio + 1)
        ne = layer.n_hidden - ni

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)

        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        # Perform QR decomposition of the matrix
        _, S, _ = np.linalg.svd(W)

        W = U @ np.diag(S) @ Vh
        layer.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')


class Bias_ZerosInit:
    @staticmethod
    def init_weights(cell):
        nn.init.zeros_(cell.b)

class W_TorchInit:
    """
    Initialization replicating pytorch's initialisation approach:

    All parameters are initialsed as
        init.uniform_(self.bias, -bound, bound)
        bound = 1 / math.sqrt(fan_in)

    Where fan_in is taken to be n_hidden for rnn cells.

    Assumes an RNN cell with W, U & b parameter tensors. Note that RNNCells
    in pytorch have two bias vectors - one for h2h and one for i2h. Wherease
    this init only assumes one.

    ## Init justification:
    I think this is a "good-bug" they have kept around due to empircal
    performance.

    Todo: add / write documentation on justiication / history of this

    e.g.
    https://soumith.ch/files/20141213_gplus_nninit_discussion.htm
    https://github.com/pytorch/pytorch/issues/57109
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L44-L48

    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    """

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.W, -bound, bound)
class U_TorchInit:
    """
    See documentation for W_TorchInit
    """

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.U, -bound, bound)
        
class U_RNNInit_ColEI_Spectrum_Init:
    
    def __init__(self, dataset='MNIST'):
        if dataset == 'MNIST':
            self.pixel_mean = .1307
        elif dataset == 'KMNIST':
            self.pixel_mean = .1918
        elif dataset == 'FashionMNIST':
            self.pixel_mean = .2860
        elif dataset == 'PennTreebank':
            self.pixel_mean = .0
        elif dataset == 'Adding':
            self.pixel_mean = .3
        # print(dataset, self.pixel_mean)

    def init_weights(self, layer):
        # Weights

        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))
        
        _, S, _ = np.linalg.svd(U_np)
        
        temp = np.zeros((layer.n_hidden, layer.n_input))
        np.fill_diagonal(temp, S)
        S = temp
        
        bound = 1 / math.sqrt(layer.n_hidden)
        U_np = np.random.uniform(-bound, bound, size=(layer.n_hidden, layer.n_input))
        
        U, _, Vh = np.linalg.svd(U_np)
        
        U_np = U @ S @ Vh

        layer.U.data = torch.from_numpy(U_np).float()

        # bias
        z_mean = layer.n_input * sigma * self.pixel_mean
        nn.init.constant_(layer.b, val=-z_mean)
        
class U_ColEIInit:
    """
    See documentation for W_TorchInit
    """
    def __init__(self, dataset='MNIST'):
        if dataset == 'MNIST':
            self.pixel_mean = .1307
        elif dataset == 'KMNIST':
            self.pixel_mean = .1918
        elif dataset == 'FashionMNIST':
            self.pixel_mean = .2860
        elif dataset == 'PennTreebank':
            self.pixel_mean = .0
        elif dataset == 'Adding':
            self.pixel_mean = .3
        # print(dataset, self.pixel_mean)

    def init_weights(self, layer):
        # Weights

        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))

        layer.U.data = torch.from_numpy(U_np).float()

        # bias
        z_mean = layer.n_input * sigma * self.pixel_mean
        nn.init.constant_(layer.b, val=-z_mean)

#     def init_weights(self, layer):
        
#         ni = layer.n_hidden // 10
#         ne = layer.n_hidden

#         # Weights
#         # print(layer.n_hidden, layer.n_input)
        
#         denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))
#         # denom = (n_excitation + n_inhibition)

#         sigma_e = np.sqrt(1/denom)
        
#         # sigma_e = np.sqrt(1/layer.n_input)
#         sigma_i = sigma_e * (ne/ni)
#         Ue_np = np.random.exponential(scale=sigma_e, size=(layer.n_hidden, ne))
#         Ui_np = -np.random.exponential(scale=sigma_i, size=(layer.n_hidden, ni))
#         U_np= np.concatenate([Ue_np, Ui_np], axis=1)
#         layer.U_pos.data = torch.from_numpy(U_np).float()

#         # bias
#         nn.init.zeros_(layer.b)

class Dense_ColEIInit_RNN_Spectrum_Init:
    
    def __init__(self, ratio=9, num=2):
        self.ratio = 9
        self.num = num
    
    def init_weights(self, layer):
        """
        todo
        """
        ni = layer.n_input // (self.ratio + 1)
        ne = layer.n_input - ni

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_output, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_output, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
#         if self.ablate_ii: W[-ni:,-ni:] = 0

        U, _, Vh = np.linalg.svd(W)

        # generate pytorch init
        bound = 1 / math.sqrt(layer.n_input)
        W = np.random.normal(loc=0.0, scale=np.sqrt(self.num/layer.n_input), size=(layer.n_output, layer.n_input))
        _, S, _ = np.linalg.svd(W)
        
        temp = np.zeros((layer.n_output, layer.n_input))
        np.fill_diagonal(temp, S)
        S = temp

        W = U @ S @ Vh
        layer.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')

class W_ColEIInit_RNN_Spectrum:
    
    def __init__(self, ratio=9, spectral_radius=1.5):
        self.ratio = 9
        self.spectral_radius = spectral_radius
    
    def init_weights(self, layer):
        """
        todo
        """
        ni = layer.n_hidden // (self.ratio + 1)
        ne = layer.n_hidden - ni

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
#         if self.zero_diag: np.fill_diagonal(W, 0)
#         if self.ablate_ii: W[-ni:,-ni:] = 0
        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        U, _, Vh = np.linalg.svd(W)

        # generate pytorch init
        bound = 1 / math.sqrt(layer.n_hidden)
        W = np.random.uniform(-bound, bound, size=(layer.n_hidden,layer.n_hidden))
        _, S, _ = np.linalg.svd(W)
        

        W = U @ np.diag(S) @ Vh
        layer.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')
    
class U_ColEIInit_RNN_Spectrum_Init:
    
    def __init__(self, ratio=9):
        self.ratio = 9
    
    def init_weights(self, layer):
        """
        todo
        """
        # Pytorch init
        bound = 1 / math.sqrt(layer.n_hidden)
        W = np.random.uniform(-bound, bound, size=(layer.n_hidden,layer.n_input))
        _, S, _ = np.linalg.svd(W)
        
        temp = np.zeros((layer.n_hidden, layer.n_input))
        np.fill_diagonal(temp, S)
        S = temp
        
        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))

        U, _, Vh = np.linalg.svd(U_np)


        W = U @ S @ Vh
        layer.U.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')
        
class Bias_TorchInit:
    """
    See documentation for W_TorchInit
    """

    def init_weights(self, cell):
        bound = 1 / math.sqrt(cell.n_hidden)
        nn.init.uniform_(cell.b, -bound, bound)

# ------------- Hidden State Initialization ----------

class Hidden_ZerosInit(nn.Module):
    def __init__(self, n_hidden, requires_grad=False):
        """
        Class to reset hidden state, for example between batches.
        To learn this initial hidden state, pass requires_grad = True.
        If requires_grad = False, hidden state will always be reset back to 0s.
        """
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(n_hidden, 1), requires_grad)

    #         print(self.hidden_init.shape)

    def reset(self, cell, batch_size):
        # print("Hidden_ZerosInit",batch_size)
        cell.h = self.h0.repeat(1, batch_size)  # Repeat tensor along bath dim.

class EiRNNCell_WeightInitPolicy():
    """
    From Pingsheng: This class is not used or updated anymore.

    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.

    Todo - update this with the different ideas
    """
    def __init__(self,numerator=1/3):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.numerator = numerator
    
    def init_weights(self,layer):
        # first for U
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Uex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))

        if layer.ni == 1: # for example the output layer
            Uix_np = Uex_np.mean(axis=0, keepdims=True)  # not random as only one int
            Uei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni

        elif layer.ni != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Uix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.n_input))
            Uei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni
        else:
            Uix_np, Uei_np = None, None
            raise ValueError('Invalid value for layer.ni, should be a positive integer.')

        layer.Uex.data = torch.from_numpy(Uex_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Uix.data = torch.from_numpy(Uix_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Uei.data = torch.from_numpy(Uei_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')

        # now for W
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.ne))

        if layer.ni == 1: # for example the output layer
            Wix_np = Wex_np.mean(axis=0,keepdims=True) # not random as only one int
            Wei_np = np.ones(shape = (layer.ne, layer.ni))/layer.ni

        elif layer.ni != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.ne))
            Wei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni
        else:
            Wix_np, Wei_np = None, None
            raise ValueError('Invalid value for layer.ni, should be a positive integer.')

        layer.Wex.data = torch.from_numpy(Wex_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Wix.data = torch.from_numpy(Wix_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        layer.Wei.data = torch.from_numpy(Wei_np).float().to('cuda' if torch.cuda.is_available() else 'cpu')

        # finally bias
        nn.init.zeros_(layer.b)

class EiRNNCell_W_InitPolicy():
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.

    Todo - update this with the different ideas
    """
    def __init__(self, numerator=1/3, distribution="exponential", centered=False, random=True):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.numerator = numerator
        self.distribution = distribution
        self.centered = centered
        self.random = random
    
    def init_weights(self,layer):
        # for W
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        # target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.ne)))

        if self.distribution == 'exponential':
            exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
            Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.ne))

            if layer.ni_h2h == 1: # for example the output layer
                if not self.random: Wix_np = Wex_np.mean(axis=0,keepdims=True) # not random as only one int
                else: Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni_h2h, layer.ne))
                # Wei_np = np.ones(shape = (layer.ne, layer.ni_h2h))/layer.ni_h2h
                Wei_np = np.random.exponential(scale=exp_scale*np.sqrt(layer.ne/layer.ni_h2h), size=(layer.ne, layer.ni_h2h))
                Wei_np /= Wei_np.sum(axis=1, keepdims=True)

            elif layer.ni_h2h != 1:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                if not self.random: Wix_np = np.ones(shape=(layer.ni_h2h,1)) * Wex_np.mean(axis=0,keepdims=True)
                else: Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni_h2h, layer.ne))
                Wei_np = np.random.exponential(scale=exp_scale*np.sqrt(layer.ne/layer.ni_h2h), size=(layer.ne, layer.ni_h2h))
                Wei_np /= Wei_np.sum(axis=1, keepdims=True)
                # Wei_np = np.ones(shape=(layer.ne, layer.ni_h2h))/layer.ni_h2h
            else:
                Wix_np, Wei_np = None, None
                raise ValueError('Invalid value for layer.ni, should be a positive integer.')
        elif self.distribution == "lognormal":
            if not self.centered: mu, sigma = calc_ln_mu_sigma(target_std, (target_std)**2)
            else: mu, sigma = calc_ln_mu_sigma(np.sqrt(0.5+np.sqrt(target_std**2+0.25)), (target_std)**2) # median will be 1
            Wex_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.ne))

            if layer.ni_h2h == 1: # for example the output layer
                if not self.random: Wix_np = Wex_np.mean(axis=0,keepdims=True) # not random as only one int
                else: Wix_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ni_h2h, layer.ne))
                # Wei_np = np.ones(shape = (layer.ne, layer.ni_h2h))/layer.ni_h2h
                Wei_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.ni_h2h))
                Wei_np /= Wei_np.sum(axis=1, keepdims=True)

            elif layer.ni_h2h != 1:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                mu, sigma = calc_ln_mu_sigma(target_std*np.sqrt(layer.ne/layer.ni_h2h), (target_std*np.sqrt(layer.ne/layer.ni_h2h))**2)
                if not self.random: Wix_np = np.ones(shape=(layer.ni_h2h,1)) * Wex_np.mean(axis=0,keepdims=True)
                else: Wix_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ni_h2h, layer.ne))
                Wei_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.ni_h2h))
                Wei_np /= Wei_np.sum(axis=1, keepdims=True)
            else:
                Wix_np, Wei_np = None, None
                raise ValueError('Invalid value for layer.ni, should be a positive integer.')
        else: 
            raise ValueError('Invalid distribution, should be lognormal or exponential.')

        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()


class EiRNNCell_W_HybridInit():
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.

    Todo - update this with the different ideas
    """
    def __init__(self, p=0.25, shift=0, numerator=1/3, distribution="exponential", centered=False):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.p = p
        self.shift = shift
        self.numerator = numerator
        self.distribution = distribution
        self.centered = centered
    
    def init_weights(self,layer):
        # for W
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        target_std = np.sqrt( (layer.ne/((layer.ne-1))  * (self.numerator/layer.n_input)))

        if self.distribution == 'exponential':
            exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
            Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.ne))

            if layer.ni_h2h == 1: # for example the output layer
                #Wix_np = np.sqrt(self.p)*Wex_np.mean(axis=0,keepdims=True) # not random as only one int
                Wix_np = np.sqrt(self.p)*np.random.exponential(scale=exp_scale, size=(layer.ni_h2h, layer.ne))
                Wei_np = np.sqrt(self.p)*np.ones(shape = (layer.ne, layer.ni_h2h))/layer.ni_h2h

            elif layer.ni_h2h != 1:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                # Wix_np = np.sqrt(self.p)*np.ones(shape=(layer.ni_h2h,1)) * Wex_np.mean(axis=0,keepdims=True)
                Wix_np = np.sqrt(self.p)*np.random.exponential(scale=exp_scale, size=(layer.ni_h2h, layer.ne))
                Wei_np = np.random.exponential(scale=exp_scale*np.sqrt(layer.ne/layer.ni_h2h), size=(layer.ne, layer.ni_h2h))
                Wei_np /= Wei_np.sum(axis=1, keepdims=True)/np.sqrt(self.p)
                # Wei_np = np.ones(shape=(layer.ne, layer.ni_h2h))/layer.ni_h2h
            else:
                Wix_np, Wei_np = None, None
                raise ValueError('Invalid value for layer.ni, should be a positive integer.')
            Wex_np *= self.p
            Wex_np += np.roll((1-self.p)*np.identity(layer.ne), self.shift, axis=(1,0))
            
        elif self.distribution == "lognormal":
            if not self.centered: mu, sigma = calc_ln_mu_sigma(target_std, (target_std)**2)
            else: mu, sigma = calc_ln_mu_sigma(np.sqrt(0.5+np.sqrt(target_std**2+0.25)), (target_std)**2) # median will be 1
            Wex_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.ne))

            if layer.ni_h2h == 1: # for example the output layer
                # Wix_np = np.sqrt(self.p)*Wex_np.mean(axis=0,keepdims=True) # not random as only one int
                Wix_np = np.sqrt(self.p)*np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ni_h2h, layer.ne))
                Wei_np = np.sqrt(self.p)*np.ones(shape = (layer.ne, layer.ni_h2h))/layer.ni_h2h

            elif layer.ni_h2h != 1:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                Wix_np = np.sqrt(self.p)*np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ni_h2h, layer.ne))
                mu, sigma = calc_ln_mu_sigma(target_std*np.sqrt(layer.ne/layer.ni_h2h), (target_std*np.sqrt(layer.ne/layer.ni_h2h))**2)
                # Wix_np = np.sqrt(self.p) * np.ones(shape=(layer.ni_h2h,1)) * Wex_np.mean(axis=0,keepdims=True)
                Wei_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.ni_h2h))
                Wei_np /= Wei_np.sum(axis=1, keepdims=True)/np.sqrt(self.p)
            else:
                Wix_np, Wei_np = None, None
                raise ValueError('Invalid value for layer.ni, should be a positive integer.')
            Wex_np *= self.p
            Wex_np += np.roll((1-self.p)*np.identity(layer.ne), self.shift, axis=(1,0))
        else: 
            raise ValueError('Invalid distribution, should be lognormal or exponential.')

        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()


class EiRNNCell_U_InitPolicy():
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.

    Todo - update this with the different ideas
    centered, if true then median = 1
    """
    def __init__(self,numerator=1/3, distribution="exponential", centered=False):
        """
        2 for he, 1/3 for pytorch, 1 for xavier
        """
        self.numerator = numerator
        self.distribution = distribution
        self.centered = centered
    
    def init_weights(self,layer):
        # for U
        # target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
        # target_std = np.sqrt( (layer.ne/((layer.ne-1)) * (self.numerator/layer.n_input)))
        target_std = np.sqrt( (layer.ne/((layer.ne-1)) * (self.numerator/layer.ne)))

        if self.distribution == 'exponential': 
            exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
            Uex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))

            if layer.ni_i2h == 1: # for example the output layer
                Uix_np = Uex_np.mean(axis=0, keepdims=True)  # not random as only one int
                Uei_np = np.ones(shape=(layer.ne, layer.ni_i2h))/layer.ni_i2h

            elif layer.ni_i2h != 1:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                Uix_np = np.ones(shape=(layer.ni_i2h,1)) * Uex_np.mean(axis=0,keepdims=True)
                Uei_np = np.random.exponential(scale=exp_scale*np.sqrt(layer.ne/layer.ni_i2h), size=(layer.ne, layer.ni_i2h))
                Uei_np /= Uei_np.sum(axis=1, keepdims=True)
            else:
                Uix_np, Uei_np = None, None
                raise ValueError('Invalid value for layer.ni, should be a positive integer.')
        elif self.distribution == "lognormal": 
            if not self.centered: mu, sigma = calc_ln_mu_sigma(target_std, (target_std)**2)
            else: mu, sigma = calc_ln_mu_sigma(np.sqrt(0.5+np.sqrt(target_std**2+0.25)), (target_std)**2) # median will be 1
            Uex_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.n_input))

            if layer.ni_i2h == 1: # for example the output layer
                Uix_np = Uex_np.mean(axis=0, keepdims=True)  # not random as only one int
                Uei_np = np.ones(shape=(layer.ne, layer.ni_i2h))/layer.ni_i2h

            elif layer.ni_i2h != 1:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                mu, sigma = calc_ln_mu_sigma(target_std*np.sqrt(layer.ne/layer.ni_i2h), (target_std*np.sqrt(layer.ne/layer.ni_i2h))**2)
                Uix_np = np.ones(shape=(layer.ni_i2h,1)) * Uex_np.mean(axis=0,keepdims=True)
                Uei_np = np.random.lognormal(mean=mu, sigma=sigma, size=(layer.ne, layer.ni_i2h))
                Uei_np /= Uei_np.sum(axis=1, keepdims=True)
            else:
                Uix_np, Uei_np = None, None
                raise ValueError('Invalid value for layer.ni, should be a positive integer.')
        else: 
            raise ValueError('Invalid distribution, should be lognormal or exponential.')

        layer.Uex.data = torch.from_numpy(Uex_np).float()
        layer.Uix.data = torch.from_numpy(Uix_np).float()
        layer.Uei.data = torch.from_numpy(Uei_np).float()


# -----------------------Song Init---------------------

def calculate_ColumnEi_layer_params(total: int, ratio):
    """
    For a ColumnEi model layer_params is a total number of units, and a ratio e.g 20, for 20:1.
    This is a util function to calculate n_e, n_i.
    Args:
        ratio : int
        total : int, total number of units (typically n_input to a layer)
    """
    fraction = total / (ratio+1)
    n_i = int(np.ceil(fraction))
    n_e = int(np.floor(fraction * ratio))
    return n_e, n_i


class ColumnEiCell_W_InitPolicy:
    """
    Class to initiliase weights for column ei RNN cell.

    Positive weights are drawn from an exponential distribution.
    Use D atricices to param the e and i cells.
    """
    def __init__(self, denom=None, radius=None, zero_diag=True, ablate_ii=False):
        self.denom = denom
        self.spectral_radius = radius
        self.zero_diag = zero_diag
        self.ablate_ii = ablate_ii

    # @staticmethod
    def init_weights(self, layer):
        """
        todo
        """
        ne = layer.ne_h
        ni = layer.ni_h

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
        if self.zero_diag: np.fill_diagonal(W, 0)
        if self.ablate_ii: W[-ni:,-ni:] = 0
        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        layer.W_pos.data = torch.from_numpy(np.absolute(W)).float()

        # D matrix (last ni columns are -)
        layer.D_W.data = torch.eye(ne + ni).float()
        layer.D_W.data[:, -ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)

        
class ColumnEiCell_W_Sandwich_InitPolicy:
    """
    Class to initiliase weights for column ei RNN cell.

    Positive weights are drawn from an exponential distribution.
    Use D atricices to param the e and i cells.
    """
    def __init__(self, denom=None, radius=None, zero_diag=True, ablate_ii=False):
        self.denom = denom
        self.spectral_radius = radius
        self.zero_diag = zero_diag
        self.ablate_ii = ablate_ii

    # @staticmethod
    def init_weights(self, layer):
        """
        todo
        """
        ne = layer.ne_h
        ni = layer.ni_h

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_hidden, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(layer.n_hidden, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
        if self.zero_diag: np.fill_diagonal(W, 0)
        if self.ablate_ii: W[-ni:,-ni:] = 0
        if self.spectral_radius is not None: 
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        layer.W_pos.data = torch.from_numpy(np.absolute(W)).float()

        # D matrix (last ni columns are -)
        layer.D_W.data = torch.eye(ne + ni).float()
        layer.D_W.data[:, -ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)
    
class RNN_ColumnEi_FirstCell_U_InitPolicy:
    """
    This is the weight init that should be used for the first layer a ColumnEi RNN.

    We use the bias term to center the activations, and glorot init to set the weight variance.
        bias  <- layer.n_input * sigma * mnist_mean *-1

    Weights are drawn from an exponential distribution.
    """
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.pixel_mean = .1307
        elif dataset == 'KMNIST':
            self.pixel_mean = .1918
        elif dataset == 'FashionMNIST':
            self.pixel_mean = .2860
        elif dataset == 'PennTreebank':
            self.pixel_mean = .0
        elif dataset == 'Adding':
            self.pixel_mean = .3
        # print(dataset, self.pixel_mean)

    def init_weights(self, layer):
        # Weights

        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))

        layer.U.data = torch.from_numpy(U_np).float()

        # bias
        z_mean = layer.n_input * sigma * self.pixel_mean
        nn.init.constant_(layer.b, val=-z_mean)

class ColumnEi_FirstCell_U_InitPolicy:
    """
    This is the weight init that should be used for the first layer a ColumnEi RNN.

    We use the bias term to center the activations, and glorot init to set the weight variance.
        bias  <- layer.n_input * sigma * mnist_mean *-1

    Weights are drawn from an exponential distribution.
    """
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.pixel_mean = .1307
        elif dataset == 'KMNIST':
            self.pixel_mean = .1918
        elif dataset == 'FashionMNIST':
            self.pixel_mean = .2860
        elif dataset == 'PennTreebank':
            self.pixel_mean = .0
        elif dataset == 'Adding':
            self.pixel_mean = .3
        # print(dataset, self.pixel_mean)

    def init_weights(self, layer):
        # Weights

        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))

        layer.U_pos.data = torch.from_numpy(U_np).float()

        # D matrix (is all positive)
        layer.D_U.data = torch.eye(layer.n_input).float()

        # bias
        z_mean = layer.n_input * sigma * self.pixel_mean
        nn.init.constant_(layer.b, val=-z_mean)


class ColumnEiCell_U_InitPolicy:
    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):

        ne = layer.ne
        ni = layer.ni

        # Weights
        # print(layer.n_hidden, layer.n_input)
        
        denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))
        # denom = (n_excitation + n_inhibition)

        sigma_e = np.sqrt(1/denom)
        
        # sigma_e = np.sqrt(1/layer.n_input)
        sigma_i = sigma_e * (ne/ni)
        Ue_np = np.random.exponential(scale=sigma_e, size=(layer.n_hidden, ne))
        Ui_np = np.random.exponential(scale=sigma_i, size=(layer.n_hidden, ni))
        U_np= np.concatenate([Ue_np, Ui_np], axis=1)
        layer.U_pos.data = torch.from_numpy(U_np).float()

        layer.D_U.data = torch.eye(ne + ni).float()
        layer.D_U.data[:, -ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)
        

class ColumnEiCell_U_InitPolicy_NotBalanced:
    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):

        ne = layer.ne
        ni = layer.ni

        # Weights
        # print(layer.n_hidden, layer.n_input)
        sigma = np.sqrt(1/layer.n_input)
        U_np = np.random.exponential(scale=sigma, size=(layer.n_hidden, layer.n_input))
        layer.U_pos.data = torch.from_numpy(U_np).float()

        # D matrix (is all positive)
        layer.D_U.data = torch.eye(ne + ni).float()
        layer.D_U.data[:, -ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)
        

class ColumnEi_Dense_InitPolicy:
    
    def __init__(self, ablate_ii=False):
        self.ablate_ii = ablate_ii
    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):

        ne = layer.ne
        ni = layer.ni

        # Weights
        # print(layer.n_hidden, layer.n_input)
        # sigma_e = np.sqrt(1/layer.n_input)
        # sigma_i = sigma_e * (ne/ni)
        
        denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))
        # denom = (n_excitation + n_inhibition)

        sigma_e = np.sqrt(1/denom)
        
        # sigma_e = np.sqrt(1/layer.n_input)
        sigma_i = sigma_e * (ne/ni)
        
        We_np = np.random.exponential(scale=sigma_e, size=(layer.n_output, ne))
        Wi_np = np.random.exponential(scale=sigma_i, size=(layer.n_output, ni))
        W_np= np.concatenate([We_np, Wi_np], axis=1)
        
        layer.W_pos.data = torch.from_numpy(W_np).float()

        # D matrix (is all positive)
        layer.D_W.data = torch.eye(ne + ni).float()
        layer.D_W.data[:, -ni:] *= -1
        
        if self.ablate_ii: layer.W_pos.data[-ni:,-ni:] = 0

        # bias
        nn.init.zeros_(layer.b)


class ColumnEi_Dense_InitPolicy_NotBalance:
    """
    Weights are drawn from an exponential distribution.
    """
    def init_weights(self, layer):

        ne = layer.ne
        ni = layer.ni

        # Weights
        # print(layer.n_hidden, layer.n_input)
        sigma = np.sqrt(1/layer.n_input)
        W_np = np.random.exponential(scale=sigma, size=(layer.n_output, layer.n_input))
        layer.W_pos.data = torch.from_numpy(W_np).float()

        # D matrix (is all positive)
        layer.D_W.data = torch.eye(ne + ni).float()
        layer.D_W.data[:, -ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)



class Hidden_ZerosInit(nn.Module):
    def __init__(self, n_hidden, requires_grad=False):
        """
        Class to reset hidden state, for example between batches.
        To learn this initial hidden state, pass requires_grad = True.
        If requires_grad = False, hidden state will always be reset back to 0s.
        """
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(n_hidden, 1), requires_grad)

    #         print(self.hidden_init.shape)

    def reset(self, cell, batch_size):
        # print("Hidden_ZerosInit",batch_size)
        cell.h = self.h0.repeat(1, batch_size)  # Repeat tensor along bath dim.

# -------------------------------

class EiRNNCellWithShunt_WeightInitPolicy(EiRNNCell_WeightInitPolicy):
    def init_weights(self, layer):
        super().init_weights(layer)
        # todo
#         a_numpy = np.sqrt((2*np.pi-1)/layer.n_input) * np.ones(shape=layer.alpha.shape)
#         a = torch.from_numpy(a_numpy)
#         alpha_val = torch.log(a)
#         layer.alpha.data = alpha_val.float()

if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass
