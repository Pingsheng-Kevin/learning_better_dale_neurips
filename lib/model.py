#export

from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# export
class Model(nn.Module):
    """
    Class representing sequential model
    
    E.g a stack of RNNCells, a CNN or just an mlp.

    """
    def __init__(self, module_list=None):
        """
        Args:
            module_list: optional, a list of modules to be added to the model's ModuleDict
                         If None, then the model should be created sequentially with the
                         append method.
        """
        super().__init__()
        self.module_dict = nn.ModuleDict()  # ordered dict that respects insertion order

        if module_list is not None:
            for i, module in enumerate(module_list):
                key = ''+str(i)
                self.module_dict[key] = module
                self.module_dict[key].network_index = i
                self.module_dict[key].network_key = key

            self.__dict__.update(self.module_dict) # enable access with dot notation

    def append(self, module, key=None):
        """
        Appends a layer/ cell to the model

        Args:
            module (object) : module to be appended to the model's ModuleDict
            key (str)   : The key to associate with the module in the ModuleDict, e.g 'fc1'
                        If None, this is automatically generated from the module.__name__
                        and current size of ModuleDict
                        :param layer:
        """

        if key is None:
            key = module.__class__.__name__ + '_' + str(len(self.module_dict))

        self.module_dict[key] = module
        # Update self.__dict__ to access layer with dot notation
        self.__dict__.update(self.module_dict)

    def forward(self, x):
        for key, module in self.module_dict.items():
            try:
                x = module.forward(x)
            except: # for debugging! 
                print(key, module.__class__.__name__)
                print("x", x.shape)
                pprint([f'{name}: {p.shape}' for name, p in module.named_parameters()])
                x = module.forward(x)
        return x

    def update(self, **kwargs):
        for key, module in self.module_dict.items():
            if hasattr(module, "update"): 
                module.update(**kwargs)

    def init_weights(self, **kwargs):
        for key, module in self.module_dict.items():
            if hasattr(module, "init_weights"): 
                module.init_weights(**kwargs)

    def reset_hidden(self, batch_size, **kwargs):
        for key, module in self.module_dict.items():
            if hasattr(module, 'reset_hidden'):
                module.reset_hidden(batch_size, **kwargs)

    @property
    def n_input(self):
        return self[0].n_input

    def extra_repr(self):
        return ''

    def __getitem__(self, item):
        """Enables layers to be indexed"""
        if isinstance(item, slice):
            print("Slicing not tested yet")
            layers = []
            for i in slice:
                layers.append(list(self.module_dict)[item])
            return layers
        key = list(self.module_dict)[item]
        return self.module_dict[key]

    def __len__(self):
        return len(list(self.module_dict))
        
# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass