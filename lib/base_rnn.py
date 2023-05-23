# export
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

from lib.dense_layers import DenseLayer
from lib.model import Model

from lib import utils
from lib import rnn_basic_tasks
from lib.update_policies import SGD
from lib.init_policies import W_NormalInit, U_NormalInit, Bias_ZerosInit, Hidden_ZerosInit

class BaseRNNCell(nn.Module):
    """
    Class implementing / formalising the expected structure of our RNNCells

    This code is a bit different from "normal" because we will create
    rnn cells by composing "policies" objects for parameter init and updates etc.
    """

    def __init__(self):
        super().__init__()
        self.n_input = None
        self.n_hidden = None
        # self.network_index = None # 0 being first cell in stack
        self.nonlinearity = None  # comment what this should be
        # This should be a function in torch.nn.functional? (activation func)

        self.i2h_init_policy = None
        self.h2h_init_policy = None
        self.hidden_reset_policy = None
        self.update_policy = None

    def init_weights(self, **args):
        self.i2h_init_policy.init_weights(self, **args)
        self.h2h_init_policy.init_weights(self, **args)
        self.bias_init_policy.init_weights(self, **args)  # before was nn.init.zeros_(self.b)

    def update(self, **args):
        # update policies should each be responsible for torch.no_grad
        self.update_policy.update(self, **args)

    def reset_hidden(self, batch_size, **args):
        self.hidden_reset_policy.reset(self, batch_size, **args)

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        r = ''
        # r += str(self.__class__.__name__)+' '
        for key, param in self.named_parameters():
            r += key + ' ' + str(list(param.shape)) + ' '
        return r

class RNNCell(BaseRNNCell):
    """
    Class representing a standard RNN cell:

        U : input weights of shape n_hidden x n_input
        W : recurrent weights of shape n_hidden x n_hidden

        h_t = f(Ux + Wht-1 + b)

    This code is a bit different from "normal" because we create
    the object by compose parameter init and update "policies".
    """

    def __init__(self, n_input, n_hidden, nonlinearity=None, learn_hidden_init=False,
                 i2h_init_policy=U_NormalInit(),
                 h2h_init_policy=W_NormalInit(),
                 bias_init_policy=Bias_ZerosInit(),
                 update_policy=SGD()):
        """
        n_input  : input dimensionality
        n_hidden : self hidden state dimensionality
        etc...
        """
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        # self.network_index = None # 0 being first cell in stack
        self.U = nn.Parameter(torch.randn(n_hidden, n_input))
        self.W = nn.Parameter(torch.randn(n_hidden, n_hidden))
        self.b = nn.Parameter(torch.ones(n_hidden, 1))
        self.nonlinearity = nonlinearity

        # based on the nonlinearity switch the denominator here? basically if relu
        self.i2h_init_policy = i2h_init_policy
        self.h2h_init_policy = h2h_init_policy
        self.bias_init_policy = bias_init_policy
        self.hidden_reset_policy = Hidden_ZerosInit(n_hidden, requires_grad=learn_hidden_init)
        self.update_policy = update_policy

        self.init_weights()

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        # h x bs = (h x input *  input x bs) + (h x h * h x bs) + h
        self.z = torch.mm(self.U, x.T) + torch.mm(self.W, self.h) + self.b
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h.T

# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass
