"""
This file contains "update_policies" for updating model params.

As this can get complicated for DANNs we are using the following 
multiple inheritance approach:

    class Parent1:
        def update(self):
            print("parent1")
    
    class Parent2:
        def update(self):
            print("parent2")
        
    class Child1(Parent1, Parent2):
        def __init__(self):
            super().__init__()

        def update(self):f
            # [1:-1] to skip self class and object class
            for mixin in self.__class__.__mro__[1:-1]:
                mixin.update(self)

Where child1 is the final update policy, composed of different parents.
The order in which child1 inherits from the parents dictates the form of the
update (e.g cSGD before or after grad clipping) so be careful!

See e.g the DalesANN_cSGD_UpdatePolicy, and demo at the bottom of this file. 
"""
from errno import ELNRNG
from functools import lru_cache
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from lib import utils

# ------------ General ------------

class ClipGradNorm_Mixin():
    def __init__(self, max_grad_norm=None, **kwargs):
        self.max_grad_norm = max_grad_norm
        self.gn_counter = 0 # for counting how many times grads were scaled

    def extra_repr(self):
        r = super().extra_repr()
        r += f'max_grad_norm: {self.max_grad_norm}'
        return r

    def update(self, layer, max_grad_norm=None, *args, **kwargs):
        """
        Constrains the norm of the layer params gradient to be within a certain norm.

        Args:
            max_grad_norm : if 0 or None, there is no clipping. It is intended max_grad_norm is
                           set at intialisation, but it can also be passed in here.

        reference implementation:
        https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
        """
        if max_grad_norm is not None:
            self.max_grad_norm =max_grad_norm

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            device = utils.get_device()
            parameters = [p for p in layer.parameters() if p.grad is not None]           

            p_norms = [torch.norm(p.grad.detach(), p='fro').to(device) for p in parameters]
            total_norm = torch.norm(torch.stack(p_norms), p='fro')
            #print(total_norm)
            clip_coef = self.max_grad_norm / (total_norm + 1e-12)
            if clip_coef < 1:
                #clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                self.gn_counter +=1
                for p in parameters:
                    # tensor.detach() creates a tensor with requires_grad=False and shares storage with original tensor
                    # mul_ then modfies this tensor inplace, changing the storage of original grad
                    p.grad.detach().mul_(clip_coef.to(p.grad.device))

        #super().update(layer, *args, **kwargs)
class SGD():
    def __init__(self, **kwargs):
        "required to allow arguments to other mixins"
        pass

    def update(self, layer, **kwargs):
        """
        Args:
            lr : learning rate
        """
        lr = kwargs['lr']
        with torch.no_grad():
            for key, p in layer.named_parameters():
                if p.requires_grad:
                    p -= p.grad * lr
                    
                    
# ------------ Dense Layer Specific ------------

class BaseUpdatePolicy():
    """
    Generic update policy.  Update is not a static method as we may want grad history
    etc for some update policies.
    """
    def __init__(self, **kwargs):
        for mixin in self.__class__.__mro__[2:-1]:
            mixin.__init__(self, **kwargs)
            
    def update(self, layer, **kwargs):
        # skip self class and object class
        for mixin in self.__class__.__mro__[2:-1]:
            mixin.update(self, layer, **kwargs)
        
    def extra_repr(self):
        """
        Reimplement this method to print out any further details
        """
        r = super().extra_repr()
        return r 
        
class EiDense_UpdatePolicy():
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g
    to be positive.

    It assumes the layer has the parameters of an EiDenseWithShunt layer.

    This should furthest right when composed with mixins for correct MRO.
    '''
    def __init__(self) -> None:
        print("""EiDense_UpdatePolicy is Deprecated, and doesn't contain
        the correction either please use 
        (BaseUpdatePolicy, cSGD_Mixin, SGD, EiDense_clamp_mixin) 
        for DANN mlps""")

    def update(self, layer, **kwargs):
        """
        Args:
            lr : learning rate
        """
        lr = kwargs['lr']
        with torch.no_grad():
            if hasattr(layer, 'g'):
                layer.g -= layer.g.grad *lr
            if layer.b.requires_grad:
                layer.b -= layer.b.grad *lr
            layer.Wex -= layer.Wex.grad * lr
            layer.Wei -= layer.Wei.grad * lr
            layer.Wix -= layer.Wix.grad * lr
            if hasattr(layer, 'alpha'):
                layer.alpha -= layer.alpha.grad *lr

            layer.Wix.data = torch.clamp(layer.Wix, min=0)
            layer.Wex.data = torch.clamp(layer.Wex, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            if hasattr(layer, 'g'):
                layer.g.data   = torch.clamp(layer.g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()
class EiDense_clamp_mixin():
    def __init__(self, **kwargs):
        "required to allow arguments to other mixins"
        pass
    def update(self, layer, **kwargs):
        layer.Wix.data = torch.clamp(layer.Wix, min=0)
        layer.Wex.data = torch.clamp(layer.Wex, min=0)
        layer.Wei.data = torch.clamp(layer.Wei, min=0)
        if hasattr(layer, 'g'):
            layer.g.data   = torch.clamp(layer.g, min=0)

class WClampMixin():
    def __init__(self,**kwargs):
        """
        Clamps weight tensor to original signs.
        """
        self.W_init_signs = None
    
    def set_W_init_signs(self,layer,lr):
        print("calculating W init signs, lr", lr, layer)
        with torch.no_grad():
            # undoing the first update to get pre gd update signs
            W_init = layer.W + lr*layer.W.grad
            self.W_init_signs = torch.sign(W_init)
            print(torch.sum(self.W_init_signs))

    def update(self, layer, **kwargs):
        if self.W_init_signs is None:
            lr = kwargs['lr']
            self.set_W_init_signs(layer, lr)
        sign_mask = self.W_init_signs
        layer.W.data = torch.clamp(layer.W*sign_mask, min=1e-9)*sign_mask 
        #layer.b.data *=0

class GeneralClampMixin():
    def __init__(self, param_list, **kwargs):
        """
        Clamps weight to their original signs.
        
        Args:
            - param_list : contains a list of strings corresponding to the 
            parameter attributes we want to clamp

        WARNING: Not sure if this is WORKING, KEEPING FOR NOW AS MIGHT COME BACK TO
        """
        self.param_list = param_list
        self.p_sign_dict = {}
    
    def build_p_sign_dict(self, layer,lr):
        with torch.no_grad():
            for p_str in self.param_list:
                attr = getattr(layer, p_str) 
                attr += lr*attr.grad
                print(attr.shape)
                self.p_sign_dict[p_str] = torch.sign(attr)

    def update(self, layer, **kwargs):
        if len(self.p_sign_dict.keys()) == 0:
            lr = kwargs['lr']
            self.build_p_sign_dict(layer, lr)

        for p_str in self.param_list:
            attr = getattr(layer, p_str)
            sign_mask = self.p_sign_dict[p_str]
            clamped_attr = torch.clamp(attr*sign_mask, min=0)*sign_mask 
            setattr(layer, p_str+'.data', torch.Tensor(clamped_attr))
            print(torch.sum(torch.sign(attr)))
            print(torch.sum(torch.sign(clamped_attr)))
           
class cSGD_Mixin():
    """
    This mixin apples the inhibitory parameter update corrections
    as derived in the ICLR for "corrected SGD".

    Again assumes an EiDenseWithShunt layer.
    """
    def __init__(self, csgd_inplace=False, **kwargs):
        self.csgd_inplace = csgd_inplace

    def update(self, layer, **args):
        with torch.no_grad():
            if self.csgd_inplace:
                layer.Wix.grad.mul_(1/np.sqrt(layer.ne))
                layer.Wei.grad.mul_(1/layer.n_input)
                if hasattr(layer, 'alpha'):
                    layer.alpha.grad.mul_(1/ (np.sqrt(layer.ne)*layer.n_input))
            else: # by default not inplace
                layer.Wix.grad =  layer.Wix.grad / np.sqrt(layer.ne)
                layer.Wei.grad =  layer.Wei.grad / layer.n_input
                if hasattr(layer, 'alpha'):
                    layer.alpha.grad =  layer.alpha.grad / (np.sqrt(layer.ne)*layer.n_input)
                    
class cSGD_v2_Mixin():
    """
    This mixin apples the inhibitory parameter update corrections
    Inspired by what's derived in ICLR

    Again assumes an EiDenseWithShunt layer.
    """
    def __init__(self, csgd_inplace=False, **kwargs):
        self.csgd_inplace = csgd_inplace

    def update(self, layer, **args):
        with torch.no_grad():
            if self.csgd_inplace:
                layer.Wix.grad.mul_(1/layer.ne)
                layer.Wei.grad.mul_(1/layer.n_input)
                if hasattr(layer, 'alpha'):
                    layer.alpha.grad.mul_(1/ (layer.ne*layer.n_input))
            else: # by default not inplace
                layer.Wix.grad =  layer.Wix.grad / layer.ne
                layer.Wei.grad =  layer.Wei.grad / layer.n_input
                if hasattr(layer, 'alpha'):
                    layer.alpha.grad =  layer.alpha.grad / (layer.ne*layer.n_input)
                    

# ------------ CNN specific update policies ------------

class Conv_cSGD_Mixin():
    "DANN update corrections for convolutional network"
    def __init__(self, csgd_inplace=False, **kwargs):
        self.csgd_inplace = csgd_inplace
    
    def update(self, layer, **kwargs):
        """
        This is the same as the MLP corrections however different dims
             - ne is output channels of (econv)
             - n_input, or d is from kernel size etc
        #Todo decide on the inplace or not. 
        """
        d  = layer.d
        ne = layer.e_conv.out_channels
        
        with torch.no_grad():
            if self.csgd_inplace:
                layer.i_conv.weight.grad.mul_(1/ np.sqrt(ne))
                layer.Wei.grad.mul_(1/ d)
                layer.alpha.grad.mul_(1/ (np.sqrt(ne)* d))
                pass
            else:
                layer.i_conv.weight.grad = layer.i_conv.weight.grad / np.sqrt(ne)
                layer.Wei.grad =  layer.Wei.grad / d
                layer.alpha.grad =  layer.alpha.grad / (np.sqrt(ne)*d)
                pass
                        
class DalesCNN_SGD_UpdatePolicy():
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g.
    Here Wex, Wix are the e and i convolution filter banks respectively. 
    
    This should be inherited on the furthest right for correct MRO. 
    
    BaseUpdatePolicy just inherits nn.Module at the moment.  
    '''
    def __init__(self) -> None:
        print("""DalesCNN_SGD_UpdatePolicy is Deprecated, and doesn't contain
        the correction either please use 
        (BaseUpdatePolicy, Conv_cSGD_Mixin, SGD, DalesCNN_weight_clamp_mixin) 
        for DANN convs""")

    def update(self, layer, **args):
        """
        Args:
            lr : learning rate
        """
        lr = args['lr']
        with torch.no_grad():
            for key, p in layer.named_parameters():
                if p.requires_grad:
                    p -= p.grad * lr
                    
            layer.i_conv.weight.data = torch.clamp(layer.i_conv.weight, min=0)
            layer.e_conv.weight.data = torch.clamp(layer.e_conv.weight, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            layer.g.data   = torch.clamp(layer.g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()
class DalesCNN_clamp_mixin():
    def __init__(self, **kwargs):
        "required to allow arguments to other mixins"
        pass
    
    def update(self, layer, **args):
            layer.i_conv.weight.data = torch.clamp(layer.i_conv.weight, min=0)
            layer.e_conv.weight.data = torch.clamp(layer.e_conv.weight, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            layer.g.data   = torch.clamp(layer.g, min=0)


# ------------ RNN specific update policies ------------
class NormStabilizer_Mixin(SGD):
    def __init__(self, lambda_):
        self.lambda_ = lambda_  # for weighting penalty

    def update(self, layer, **args):
        # with torch.no_grad():
        # calculate the penalty
        pass

class EiRNN_UpdatePolicy():
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g
    to be positive.

    It assumes the layer has the parameters of an EiDenseWithShunt layer.

    This should furthest right when composed with mixins for correct MRO.
    '''
    def __init__(self) -> None:
        print("To refactor!, use (SGD, EiRNN_clamp_mixin) instead")

    def update(self, layer, **args):
        """
        Args:
            lr : learning rate
        """
        lr = args['lr']
        with torch.no_grad():
            layer.b -= layer.b.grad *lr
            layer.Uex -= layer.Uex.grad * lr
            layer.Uei -= layer.Uei.grad * lr
            layer.Uix -= layer.Uix.grad * lr
            layer.Wex -= layer.Wex.grad * lr
            layer.Wei -= layer.Wei.grad * lr
            layer.Wix -= layer.Wix.grad * lr
            if hasattr(layer, 'U_alpha'):
                layer.U_alpha -= layer.U_alpha.grad *lr
            if hasattr(layer, 'W_alpha'):
                layer.W_alpha -= layer.W_alpha.grad *lr
            if hasattr(layer, 'U_g'):
                layer.U_g -= layer.U_g.grad *lr
            if hasattr(layer, 'W_g'):
                layer.W_g -= layer.W_g.grad *lr

            layer.Wix.data = torch.clamp(layer.Wix, min=0)
            layer.Wex.data = torch.clamp(layer.Wex, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            layer.Uix.data = torch.clamp(layer.Uix, min=0)
            layer.Uex.data = torch.clamp(layer.Uex, min=0)
            layer.Uei.data = torch.clamp(layer.Uei, min=0)
            if hasattr(layer, 'U_g'):
                layer.U_g.data   = torch.clamp(layer.U_g, min=0)
            if hasattr(layer, 'W_g'):
                layer.W_g.data   = torch.clamp(layer.W_g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()

class EiRNNcSGD_Mixin():
    """
    This mixin apples the inhibitory parameter update corrections
    as derived in the paper for "corrected SGD".

    Again assumes an EiDenseWithShunt layer.

    Todo: Simplify this inplace behaviour - choose one
    """
    def __init__(self, csgd_inplace=False, **kwargs):
        self.csgd_inplace = csgd_inplace

    def update(self, layer, **args):
        with torch.no_grad():
            if self.csgd_inplace:
                layer.Uix.grad.mul_(1/np.sqrt(layer.ne))
                layer.Uei.grad.mul_(1/layer.n_input)
                layer.Wix.grad.mul_(1/np.sqrt(layer.ne))
                layer.Wei.grad.mul_(1/layer.ne) # "input" is previous hidden
                if hasattr(layer, 'U_alpha'):
                    layer.U_alpha.grad.mul_(1/ (np.sqrt(layer.ne)*layer.n_input))
                if hasattr(layer, 'W_alpha'):
                    layer.W_alpha.grad.mul_(1/ (np.sqrt(layer.ne)*layer.ne))

            else:
                layer.Uix.grad = layer.Uix.grad / np.sqrt(layer.ne)
                layer.Uei.grad = layer.Uei.grad / layer.n_input
                layer.Wix.grad = layer.Wix.grad / np.sqrt(layer.ne)
                layer.Wei.grad = layer.Wei.grad / layer.ne # "input" is previous hidden
                if hasattr(layer, 'U_alpha'):
                    layer.U_alpha.grad  = layer.U_alpha.grad / (np.sqrt(layer.ne)*layer.n_input)
                if hasattr(layer, 'W_alpha'):
                    layer.W_alpha.grad  = layer.W_alpha.grad /  (np.sqrt(layer.ne)*layer.ne)
                    
class EiRNNcSGD_v2_Mixin():
    """
    This mixin apples the inhibitory parameter update corrections
    as derived in the paper for "corrected SGD".

    Again assumes an EiDenseWithShunt layer.

    Todo: Simplify this inplace behaviour - choose one
    """
    def __init__(self, csgd_inplace=False, **kwargs):
        self.csgd_inplace = csgd_inplace

    def update(self, layer, **args):
        with torch.no_grad():
            if self.csgd_inplace:
                layer.Uix.grad.mul_(1/layer.ne)
                layer.Uei.grad.mul_(1/layer.n_input)
                layer.Wix.grad.mul_(1/layer.ne)
                layer.Wei.grad.mul_(1/layer.ne) # "input" is previous hidden
                if hasattr(layer, 'U_alpha'):
                    layer.U_alpha.grad.mul_(1/ (layer.ne*layer.n_input))
                if hasattr(layer, 'W_alpha'):
                    layer.W_alpha.grad.mul_(1/ (layer.ne*layer.ne))

            else:
                layer.Uix.grad = layer.Uix.grad / layer.ne
                layer.Uei.grad = layer.Uei.grad / layer.n_input
                layer.Wix.grad = layer.Wix.grad / layer.ne
                layer.Wei.grad = layer.Wei.grad / layer.ne # "input" is previous hidden
                if hasattr(layer, 'U_alpha'):
                    layer.U_alpha.grad  = layer.U_alpha.grad / (layer.ne*layer.n_input)
                if hasattr(layer, 'W_alpha'):
                    layer.W_alpha.grad  = layer.W_alpha.grad /  (layer.ne*layer.ne)

#--------------- Song SGD -----------------

class ColumnEiDenseSGD:
    def __init__(self, max=None, ablate_ii=False):
        self.max = max
        self.ablate_ii = ablate_ii
    
    @torch.no_grad()
    def update(self, layer, **args):
        b_norm = layer.b.grad.norm(2)
        W_pos_norm = layer.W_pos.grad.norm(2)
        # gradient clipping
        if self.max is not None:
            if b_norm > self.max:
                layer.b.grad *= (self.max / b_norm)
            if W_pos_norm > self.max:
                layer.W_pos.grad *= (self.max / W_pos_norm)

        lr = args['lr']
        layer.b     -= layer.b.grad *lr
        layer.W_pos -= layer.W_pos.grad * lr

        if layer.clamp:
            layer.W_pos.data = torch.clamp(layer.W_pos, min=0)
        if self.ablate_ii:
            layer.W_pos.data[-layer.ni:,-layer.ni:] = 0
            

class ColumnEiSGD:
    @torch.no_grad()
    def update(self, layer, **args):
        lr = args['lr']
        layer.b     -= layer.b.grad *lr
        layer.W_pos -= layer.W_pos.grad * lr
        layer.U_pos -= layer.U_pos.grad * lr

        if layer.clamp:
            layer.W_pos.data = torch.clamp(layer.W_pos, min=0)
            layer.U_pos.data = torch.clamp(layer.U_pos, min=0)


class ColumnEiSGD_Clip:
    def __init__(self, max=None, ablate_ii=False):
        self.max = max
        self.ablate_ii = ablate_ii

    @torch.no_grad()
    def update(self, layer, **args):
        b_norm = layer.b.grad.norm(2)
        W_pos_norm = layer.W_pos.grad.norm(2)
        U_pos_norm = layer.U_pos.grad.norm(2)
        # gradient clipping
        if self.max is not None:
            if b_norm > self.max:
                layer.b.grad *= (self.max / b_norm)
            if W_pos_norm > self.max:
                layer.W_pos.grad *= (self.max / W_pos_norm)
            if U_pos_norm > self.max:
                layer.U_pos.grad *= (self.max / U_pos_norm)
        
        lr = args['lr']
        layer.b     -= layer.b.grad *lr
        layer.W_pos -= layer.W_pos.grad * lr
        layer.U_pos -= layer.U_pos.grad * lr

        if layer.clamp:
            layer.W_pos.data = torch.clamp(layer.W_pos, min=0)
            layer.U_pos.data = torch.clamp(layer.U_pos, min=0)
        if self.ablate_ii:
            layer.W_pos.data[-layer.ni:,-layer.ni:] = 0


# -------- update policies here should be refactored out
class EiRNN_cSGD_UpdatePolicy(BaseUpdatePolicy, EiRNNcSGD_Mixin, ClipGradNorm_Mixin, EiRNN_UpdatePolicy):
    def __init__(self, max_grad_norm=None):
        super(EiRNN_cSGD_UpdatePolicy, self).__init__()
        self.max_grad_norm=max_grad_norm

# class DalesANN_homeostatic_UpdatePolicy(BaseUpdatePolicy, HomeostaticMixin,
#                                         cSGD_Mixin, EiDense_UpdatePolicy):
#     pass

class SGD_Clip_UpdatePolicy_NoClamp(BaseUpdatePolicy, ClipGradNorm_Mixin, SGD):
    def __init__(self, max_grad_norm=None):
        super(SGD_Clip_UpdatePolicy_NoClamp, self).__init__()
        self.max_grad_norm=max_grad_norm


class SGD_Clip_UpdatePolicy(BaseUpdatePolicy, ClipGradNorm_Mixin, SGD, WClampMixin):
    def __init__(self, max_grad_norm=None):
        super(SGD_Clip_UpdatePolicy, self).__init__()
        self.max_grad_norm=max_grad_norm

class DalesANN_homeostatic_UpdatePolicy(BaseUpdatePolicy, HomeostaticMixin,
                                        cSGD_Mixin, SGD, EiDense_clamp_mixin):
    pass

class DalesANN_cSGD_UpdatePolicy(BaseUpdatePolicy, cSGD_Mixin, ClipGradNorm_Mixin, EiDense_UpdatePolicy):
    def __init__(self, max_grad_norm=None):
        super(DalesANN_cSGD_UpdatePolicy, self).__init__()
        self.max_grad_norm=max_grad_norm


class DalesANN_conv_cSGD_UpdatePolicy(BaseUpdatePolicy, ConvHomeostaticMixin,
                                      Conv_cSGD_Mixin, DalesCNN_SGD_UpdatePolicy):
    pass




if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    pass
