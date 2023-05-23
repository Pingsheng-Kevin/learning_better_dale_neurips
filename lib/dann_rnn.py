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
from lib.init_policies import Bias_ZerosInit, Hidden_ZerosInit, EiRNNCell_W_InitPolicy, EiRNNCell_U_InitPolicy

#export
class EiRNNCell(BaseRNNCell):
    """
    Class modelling a DANN-RNN with subtractive feedforward
    inhibition between timesteps and layers

    h_t = f( (U^ex - U^eiU^ix)x + (W^ex - W^eiW^ix)h_t-1 + b)
    """
    def __init__(self, n_input, ne, ni_i2h, ni_h2h,
                 nonlinearity=None, learn_hidden_init=False,
                 i2h_init_policy=EiRNNCell_U_InitPolicy(),
                 h2h_init_policy=EiRNNCell_W_InitPolicy(),
                 bias_init_policy=Bias_ZerosInit(),
                 update_policy=None, U_dropconnect=0, W_dropconnect=0):
        """
        ne : number of excitatory units in the "main" hidden layer, h
        ni_i2h : number of inhib units between x_t and h_t
        ni_h2h : number of inhib units between h_t-1 and h_t

        Todo, pass in hidden reset policy too.
        """
        super().__init__()
        self.n_input = n_input
        self.n_hidden = ne
        self.ne = ne # some redundant naming going on here with n_hidden
        self.ni_i2h = ni_i2h
        self.ni_h2h = ni_h2h
        self.U_dropconnect = U_dropconnect
        self.W_dropconnect = W_dropconnect

        # to-from notation - U_post_pre and the shape is n_output x n_input
        self.Uex = nn.Parameter(torch.empty(ne,n_input))
        self.Uix = nn.Parameter(torch.empty(ni_i2h,n_input))
        self.Uei = nn.Parameter(torch.empty(ne,ni_i2h))

        self.Wex = nn.Parameter(torch.empty(ne,ne))
        self.Wix = nn.Parameter(torch.empty(ni_h2h,ne))
        self.Wei = nn.Parameter(torch.empty(ne,ni_h2h))

        self.b = nn.Parameter(torch.ones(ne, 1))
        self.nonlinearity = nonlinearity

        # based on the nonlinearity switch the denominator here? basically if relu
        self.i2h_init_policy = i2h_init_policy
        self.h2h_init_policy = h2h_init_policy
        self.bias_init_policy = bias_init_policy
        self.hidden_reset_policy = Hidden_ZerosInit(self.n_hidden, requires_grad=learn_hidden_init)
        self.update_policy = update_policy
        self.init_weights()

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    @property
    def U(self):
        return self.Uex - torch.matmul(self.Uei, self.Uix)

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        # h x bs = (h x input *  input x bs) + (h x h * h x bs) + h
        # print(self.U.shape, x.shape, self.W.shape, self.h.shape)

        if self.training == False: self.z = torch.mm(self.U, x.T) + torch.mm(self.W, self.h) + self.b
        else: 
            Uex_drop = F.dropout(self.Uex, p=self.U_dropconnect) * (1-self.U_dropconnect)
            Uix_drop = F.dropout(self.Uix , p=self.U_dropconnect) * (1-self.U_dropconnect)
            Wex_drop = F.dropout(self.Wex, p=self.W_dropconnect) * (1-self.W_dropconnect)
            Wix_drop = F.dropout(self.Wix , p=self.W_dropconnect) * (1-self.W_dropconnect)
            self.z = torch.mm(Uex_drop - torch.matmul(self.Uei, Uix_drop), x.T) + torch.mm(Wex_drop - torch.matmul(self.Wei, Wix_drop), self.h) + self.b
        if self.nonlinearity is not None: self.h = self.nonlinearity(self.z)
        else: self.h = self.z
        return self.h.T

# export
class EiRNNCellWithShunt(EiRNNCell):
    """
    Class modelling a DANN-RNN with subtractive and divisive
    feedforward inhibition between timesteps and layers

    Todo: update equation here
    h_t = f( (U^ex - U^eiU^ix)x + (W^ex - W^eiW^ix)h_t-1 + b)

    Todo: read up on the details of BN/LN across layers vs time,
    I remeber the first paper only managed to apply over depth (i think)
    for some reason.
    """
    def __init__(self, n_input, ne, ni_i2h, ni_h2h,
                 nonlinearity=None, learn_hidden_init=False,
                 i2h_init_policy=EiRNNCell_U_InitPolicy(),
                 h2h_init_policy=EiRNNCell_W_InitPolicy(),
                 bias_init_policy=Bias_ZerosInit(),
                 update_policy=None):
        """
        ne : number of excitatory units in the "main" hidden layer, h
        ni_i2h : number of inhib units between x_t and h_t
        ni_h2h : number of inhib units between h_t-1 and h_t
        """

        super().__init__(n_input, ne, ni_i2h, ni_h2h,
                        nonlinearity,learn_hidden_init, i2h_init_policy,
                        h2h_init_policy,bias_init_policy,update_policy)

        self.U_alpha = nn.Parameter(torch.ones(size=(1, ni_i2h))) # row vector
        self.W_alpha = nn.Parameter(torch.ones(size=(1, ni_h2h)))
        self.U_g     = nn.Parameter(torch.ones(size=(ne,1)))
        self.W_g     = nn.Parameter(torch.ones(size=(ne,1)))

        self.epsilon = 1e-8

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        self.x = x.T

        self.W_ze = self.Wex @ self.h  # ne x batch
        self.W_zi = self.Wix @ self.h  # ni x btch
        # ne x batch = ne x batch - nexni ni x batch
        self.W_z_hat = self.W_ze - self.Wei @ self.W_zi
        self.W_exp_alpha = torch.exp(self.W_alpha)  # 1 x ni
        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.W_gamma = ((self.W_exp_alpha * self.Wei) @ self.W_zi) + self.epsilon
        # ne x batch = ne x batch * ne x batch
        self.W_z = (1 / self.W_gamma) * self.W_z_hat
        # ne x batch = nex1*ne x batch + nex1
        self.W_z = self.W_g * self.W_z


        self.U_ze = self.Uex @ self.x  # ne x batch
        self.U_zi = self.Uix @ self.x  # ni x btch
        # ne x batch = ne x batch - nexni ni x batch
        self.U_z_hat = self.U_ze - self.Uei @ self.U_zi
        self.U_exp_alpha = torch.exp(self.U_alpha)  # 1 x ni
        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.U_gamma = ((self.U_exp_alpha * self.Uei) @ self.U_zi) + self.epsilon
        # ne x batch = ne x batch * ne x batch
        self.U_z = (1 / self.U_gamma) * self.U_z_hat
        # ne x batch = nex1*ne x batch + nex1
        self.U_z = self.U_g * self.U_z

        self.z = self.U_z + self.W_z + self.b
        # self.z =  g*(self.U_z + self.W_z) + self.b

        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z.clone()

        # retaining grad for ngd calculations
        if self.W_zi.requires_grad or self.U_zi.requires_grad:
            self.W_zi.retain_grad()
            self.U_zi.retain_grad()
            self.z.retain_grad()
            self.W_gamma.retain_grad()
            self.U_gamma.retain_grad()

        return self.h.T

# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass