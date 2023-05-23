#export
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
from lib.base_rnn import BaseRNNCell
from lib.init_policies import Bias_ZerosInit, Hidden_ZerosInit, ColumnEiCell_U_InitPolicy, ColumnEiCell_W_InitPolicy, calculate_ColumnEi_layer_params
from lib.update_policies import ColumnEiSGD, ColumnEiSGD_Clip

# export

# ------------- ColumnEi Cell ----------
class ColumnEiCell(BaseRNNCell):
    """
    Class representing a ColumnEi RNN cell:
    """
    def __init__(self, n_input, layer_params: '(n_hidden, ratio)', nonlinearity=None, learn_hidden_init=False,
                 i2h_init_policy=ColumnEiCell_U_InitPolicy(),
                 h2h_init_policy=ColumnEiCell_W_InitPolicy(),
                 bias_init_policy=Bias_ZerosInit(),
                 update_policy=ColumnEiSGD(),
                 clamp=True):
        """
        layer_params: (tuple) (ouput, ratio of e (to i))
        """
        # print(n_input, layer_params)

        super().__init__()
        n_hidden, ratio = layer_params
        self.ne, self.ni = calculate_ColumnEi_layer_params(n_input, ratio)
        self.ne_h, self.ni_h = calculate_ColumnEi_layer_params(n_hidden, ratio)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.nonlinearity = nonlinearity
        self.U_pos = nn.Parameter(torch.randn(n_hidden, n_input))
        self.D_U = nn.Parameter(torch.empty(self.n_input, self.n_input), requires_grad=False)
        self.W_pos = nn.Parameter(torch.randn(n_hidden, n_hidden))
        self.D_W = nn.Parameter(torch.empty(self.n_hidden, self.n_hidden), requires_grad=False)
        self.b = nn.Parameter(torch.ones(n_hidden, 1))

        self.clamp = clamp

        # based on the nonlinearity switch the denominator here? basically if relu
        self.i2h_init_policy = i2h_init_policy
        self.h2h_init_policy = h2h_init_policy
        self.bias_init_policy = bias_init_policy
        self.hidden_reset_policy = Hidden_ZerosInit(n_hidden, requires_grad=learn_hidden_init)
        self.update_policy = update_policy

        self.init_weights()

    @property
    def W(self):
        return self.W_pos @ self.D_W

    @property
    def U(self):
        return self.U_pos @ self.D_U

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        # print(self.U.shape, x.shape, self.W.shape, self.h.shape, self.b.shape)
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