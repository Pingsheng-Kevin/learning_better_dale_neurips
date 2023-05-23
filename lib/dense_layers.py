#export
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils
from lib import rnn_basic_tasks
from lib import init_policies as ip
from lib import update_policies as up
from lib.init_policies import DenseNormalInit, calculate_ColumnEi_layer_params, Bias_ZerosInit, EiDenseWeightInit_WexMean, ColumnEi_Dense_InitPolicy
from lib.update_policies import SGD, EG, ColumnEiDenseSGD

class BaseDenseLayer(nn.Module):
    """
    Base class implementing / formalising the expected structure of fully connected layer
    and how the update and weight init policies work etc.
    """
    def __init__(self):
        super().__init__()
        self.n_input = None
        self.n_output = None
        self.nonlinearity = None

        self.weight_init_policy = None
        self.update_policy = None

    @property
    def input_shape(self): return self.n_input
    @property
    def output_shape(self): return self.n_output

    def forward(self, *inputs):
        raise NotImplementedError

    def init_weights(self, **args):
        # with the EG Layers it is now unclear if biases should be init seperately
        # also might need to change this for RNNs too
        self.weight_init_policy.init_weights(self, **args)

    def update(self, **kwargs):
        # update policies should each be responsible for
        # with torch.no_grad: context.
        self.update_policy.update(self, **kwargs)

    @property
    def param_names(self):
        return [p[0] for p in self.named_parameters()]

    def __repr__(self):
        """
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L1529
        # def extra_repr(self):
        """
        r  = ''
        r += str(self.__class__.__name__)+' '
        for key, param in self.named_parameters():
            r += key +' ' + str(list(param.shape))+' '
        if self.nonlinearity is None:
            r += 'linear'
        else:
            r += str(self.nonlinearity.__name__)

        child_lines = []
        for key, module in self._modules.items():
            child_repr = "  "+repr(module)
            child_lines.append('(' + key + '): ' + child_repr)

        r += '\n  ' + '\n  '.join(child_lines) #+ '\n'
        return r

class DenseLayer(BaseDenseLayer):
    def __init__(self, n_input, n_output, nonlinearity=None,
                 weight_init_policy=DenseNormalInit(),
                 update_policy=SGD()):
        """
        n_input:      input dimension
        n_output:     output dimension
        nonlinearity (obj or None): nonlinear activation function, if None then linear
        weight_init_policy:
        update_policy:
        """
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.nonlinearity = nonlinearity
        self.weight_init_policy = weight_init_policy
        self.update_policy = update_policy

        self.W = nn.Parameter(torch.randn(n_output, n_input))
        self.b = nn.Parameter(torch.zeros(n_output, 1))

        self.init_weights()

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim ??? Why I got error
        """
        self.x = x.T
        self.z = torch.mm(self.W, self.x) + self.b
        
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h.T

class EiDense(BaseDenseLayer):
    """
    Class modeling a Feed-forward inhibition layer without shunting
    """
    def __init__(self, n_input, ne, ni, nonlinearity=None,
                 weight_init_policy=EiDenseWeightInit_WexMean(), update_policy=None):
        """
        ne : number of exciatatory outputs
        ni : number of inhibtitory units
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = ne
        self.nonlinearity = nonlinearity
        self.weight_init_policy = weight_init_policy
        self.update_policy = update_policy

        self.ne = ne
        self.ni = ni

        # to-from notation - W_post_pre and the shape is n_output x n_input
        self.Wex = nn.Parameter(torch.empty(self.ne,self.n_input))
        self.Wix = nn.Parameter(torch.empty(self.ni,self.n_input))
        self.Wei = nn.Parameter(torch.empty(self.ne,self.ni))
        self.b   = nn.Parameter(torch.zeros(self.ne,1))

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim ??? Why I got error?
        """
        self.x = x.T
        self.z = torch.matmul(self.W, self.x)
        self.z = self.z + self.b
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        # if self.z.requires_grad:
        #     self.z.retain_grad()
        #     self.h.retain_grad()
        return self.h.T
        
class EiDenseWithShunt(EiDense):
    """
    Class modeling a Feed-forward inhibition layer including a shunting
    component of inhibition
    """
    def __init__(self, n_input, ne, ni, nonlinearity=None,
                 weight_init_policy=None,update_policy=None):
        """
        ne : number of exciatatory outputs
        ni : number of inhibtitory units
        """
        super().__init__(n_input,  ne, ni, nonlinearity,
                         weight_init_policy, update_policy)
        self.alpha = nn.Parameter(torch.ones(size=(1, self.ni))) # row vector
        self.g     = nn.Parameter(torch.ones(size=(self.ne,1)))
        self.epsilon = 1e-8

    
    @property
    def alpha_property(self,x):
        """
        A filthy hack account for the new divisive inhibition schemes.
        But we would still need to decide what is a parameter (alpha or rho) in init
        """
        if hasattr(self, "rho"):
            return torch.softplus(self.rho)
        else:
            return torch.exp(self.alpha)

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim
        """
        self.x = x.T
        self.ze = self.Wex@self.x  # ne x batch
        self.zi = self.Wix@self.x  # ni x btch

        # ne x batch = ne x batch - nexni ni x batch
        self.z_hat = self.ze - self.Wei@self.zi
        self.exp_alpha = torch.exp(self.alpha) # 1 x ni

        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.gamma = ((self.exp_alpha*self.Wei)@self.zi) + self.epsilon

        # ne x batch = ne x batch * ne x batch
        self.z_dot = (1/ self.gamma) * self.z_hat

        # ne x batch = nex1*ne x batch + nex1
        self.z = self.g*self.z_dot + self.b
        # batch x ne
        self.z = self.z      # return batch to be first axis
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z.clone()

        # retaining grad for ngd calculations
        # if self.zi.requires_grad:
        #     self.zi.retain_grad()
        #     self.z.retain_grad()
        #     self.gamma.retain_grad()
        return self.h.T


class EiDenseWithShuntLayerNorm(EiDense):
    """
    Class modeling a Feed-forward inhibition layer including a shunting
    component of inhibition
    """
    def __init__(self, n_input, ne, ni, nonlinearity=None,
                 weight_init_policy=None,update_policy=None):
        """
        ne : number of exciatatory outputs
        ni : number of inhibtitory units
        """
        super().__init__(n_input,  ne, ni, nonlinearity,
                         weight_init_policy, update_policy)
        self.g     = nn.Parameter(torch.ones(size=(self.ne,1)))

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim
        """
        self.x = x.T
        self.ze = self.Wex@self.x  # ne x batch
        self.zi = self.Wix@self.x  # ni x btch

        # ne x batch = ne x batch - nexni ni x batch
        self.z_hat = self.ze - self.Wei@self.zi

        self.mu_z_layer  = self.z_hat.mean(axis=0, keepdim=True) # 1 x batch
        self.std_z_layer = self.z_hat.std(axis=0, keepdim=True) # 1 x batch

        # ne x batch = (1 / 1 x batch) * ( ne x batch - 1 x batch )
        self.z_dot = (1/ self.std_z_layer) * (self.z_hat - self.mu_z_layer)

        # ne x batch = nex1*ne x batch + nex1
        self.z = self.g*self.z_dot + self.b
        
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z.clone()

        # retaining grad for ngd calculations
        if self.zi.requires_grad:
            self.zi.retain_grad()
            self.z.retain_grad()
        return self.h.T

class ColumnEiDense(BaseDenseLayer):
    """
    Class modeling a Feed-forward inhibition layer without shunting
    """
    def __init__(self, n_input, layer_params: '(n_output, ratio)', nonlinearity=None, learn_hidden_init=False,
                weight_init_policy=ColumnEi_Dense_InitPolicy(),
                update_policy=ColumnEiDenseSGD(),
                clamp=True):
        """
        ne : number of exciatatory outputs
        ni : number of inhibtitory units
        """
        super().__init__()
        n_output, ratio = layer_params
        self.ne, self.ni = calculate_ColumnEi_layer_params(n_input, ratio)
        self.n_input = n_input
        self.n_output = n_output
        self.nonlinearity = nonlinearity
        self.W_pos = nn.Parameter(torch.randn(n_output, n_input))
        self.D_W = nn.Parameter(torch.empty(self.n_input, self.n_input), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(self.n_output, 1))
        self.weight_init_policy = weight_init_policy
        self.update_policy = update_policy
        self.clamp = clamp

    @property
    def W(self):
        return self.W_pos @ self.D_W

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim ??? Why I got error?
        """
        self.x = x.T
        self.z = torch.matmul(self.W, self.x) + self.b
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        # if self.z.requires_grad:
        #     self.z.retain_grad()
        #     self.h.retain_grad()
        return self.h.T

class EgDenseLayer(DenseLayer):
    def __init__(self, n_input, n_output, nonlinearity=None, weight_init_policy=DenseNormalInit(), update_policy=SGD()):
        super().__init__(n_input, n_output, nonlinearity, weight_init_policy, update_policy)
        self.b_pos = nn.Parameter(torch.ones(n_output,1))
        self.b_neg = nn.Parameter(torch.ones(n_output,1)*-1)
        del self.b
        
    @property
    def b(self):
        return self.b_pos + self.b_neg

class EgEiDense(EiDense):
    def __init__(self, n_input, ne, ni, nonlinearity=None, weight_init_policy=None, update_policy=None):
        super().__init__(n_input, ne, ni, nonlinearity, weight_init_policy, update_policy)
        self.b_pos = nn.Parameter(torch.ones(self.ne,1))
        self.b_neg = nn.Parameter(torch.ones(self.ne,1)*-1)
        del self.b
    
    @property
    def b(self):
        return self.b_pos + self.b_neg

class EgEiDenseWithShunt(EiDenseWithShunt):
    def __init__(self, n_input, ne, ni, nonlinearity=None, weight_init_policy=None, update_policy=None):
        super().__init__(n_input, ne, ni, nonlinearity, weight_init_policy, update_policy)
        self.b_pos = nn.Parameter(torch.ones(self.ne,1))
        self.b_neg = nn.Parameter(torch.ones(self.ne,1)*-1)
        del self.b

    @property
    def b(self):
        return self.b_pos + self.b_neg



if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    #
    l = EgEiDenseWithShunt(784, 10,1,None, ip.EiDenseWithShunt_WeightInitPolicy_ICLR, EG)
    print(l.b_pos)
    print(l.b_neg)
    #print(l.b)

    l = EgDenseLayer(784, 10)
    l.weight_init_policy = ip.DenseNormalInit()
    print(l.b)
    print(l.b_neg)
    print(l.b_pos)
    pass
