import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler

from lib import utils
from pathlib import Path


def get_torchvision_data(dataset_name, directory = str(Path.home()/ "scratch/data")) -> Path:
    """ 
    Returns path to a torchvision dataset. Downloads using torchvision if not found in data_root.
    
    Args:
        dataset_name: str - currently only MNIST, KMNIST or FashionMNIST
        directory  : str - Location where data is stored, default is "../data"
    
    Returns:
        pathlib.Path to dataset, e.g PosixPath('../data/FashionMNIST')
        
    Note: Think some of the torchivison datasets have slightly different formats, for
    example Imagenet has a "split" arg not train.
    """
    data_root = Path(directory) 
    if dataset_name == 'MNIST':
        torchvision.datasets.MNIST(root=data_root, train=True, download=True)
        torchvision.datasets.MNIST(root=data_root, train=False, download=True)
    elif dataset_name == 'KMNIST':
        torchvision.datasets.KMNIST(root=data_root, train=True, download=True)
        torchvision.datasets.KMNIST(root=data_root, train=False, download=True)
    elif dataset_name == 'FashionMNIST':
        torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True)
        torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True)
    else:
        print('Unsupported Dataset string')
        raise
    return  data_root/dataset_name

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, flatten=False, permute=False, to_device=True,
                 named_tensors=False):
        """
        Class representing MNIST-like dataset (i.e. also Kuzushiji or Fashion).
        
        Intended use is to represent full training or test sets. 
        Use torch.utils.data.sampler.SubsetRandomSampler 
        to obtain a validation set (when contructing dataloaders). 
        
        Args:
            path : path to the data.pt file,
                    e.g. '../data/MNIST/processed/training.pt'
            flatten: flattens data so rows are stacked, if true x is of
                    shape batch, n_pixels. If false x is unsqueezed to
                    batch, col (=1), height, width

            to_device: If true puts tensors on device returned by utils.get_device()
            permute: To implement
            named_tensors: ERROR! Stack is not yet supported with named tensors, so
                           this needs to be false for now. 
                          A flag to set whether we want to name the tensors. Currently
                          only for the conv models (i.e flatten is False).                
        """
        self.loadpath = path
        self.x, self.y = torch.load(path) 
        self.x = self.x.float().div(255)
        self.n_classes = len(self.y.unique())
        
        if flatten: 
            self.x = self.x.reshape(self.x.shape[0],-1).contiguous()
        else:
            # unsqueeze so we have BCHW
            self.x = self.x.unsqueeze(1).contiguous()
            
        if not flatten and named_tensors:
            # this throws error in current pytorch version 
            self.x = self.x.refine_names('batch', 'channels', 'rows', 'columns')

        if to_device:
            device = utils.get_device()
            self.x = self.x.to(device)
            self.y = self.y.to(device)
            
        if permute:
            raise 
            
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    @property
    def data_shape(self):
        # grabs first datapoint which will be (x,y) tuple so [0] again for x
        return tuple(self[0][0].shape)

    @property
    def n_pixels(self):
        return np.prod(self.data_shape)
    
    def __repr__(self):
        s = super().__repr__()
        s += '\nData loaded from: '+str(self.loadpath)
        s += '\n'
        s += f'Data is {self.__len__()} examples of shape {self.data_shape}'
        return s
    

def get_data(dataset_name, batch_size, directory=f"{os.environ['HOME']}/data", seed=7,
             to_device=False, validation_size=10000, flatten=False,
             copy_to_slurmtmpdir=True, test_set=False):
    """
    Convenience function to wrap up steps to get data. Returns two dataloaders for training and eval
    
    Args:
        dataset_name: One of [MNIST, KMNIST, FashionMNIST]
        test_set - Bool: If True, validation size is not used, full training set loader
                     and test loader are returned
        to_device - Bool: Whether to pre-push the data onto the GPU
    
    If running locally set copy_to_slurmtmpdir to false (as this won't exist). You might also want
    to set the directory to ./data, not ..

    Note: Not sure if we will use this going forward - copying and downloading to scratch vs using MILA's datasets
    """
    data_dir = get_torchvision_data(dataset_name, directory)
    if copy_to_slurmtmpdir:
        data_dir = utils.copy_folder_to_slurm_tmpdir(data_dir)
    
    train_dataset = MNISTDataset(data_dir/"processed"/"training.pt", flatten, to_device=to_device)
    if test_set:
        test_dataset = MNISTDataset(data_dir/"processed"/"test.pt", flatten, to_device=to_device)
        test_loader  = get_dataloader(test_dataset,batch_size,  shuffle=False)
        train_loader = get_dataloader(train_dataset,batch_size, shuffle=True) # Should be true
        return train_loader, test_loader
    
    else:       
        train_loader, valid_loader = get_train_val_dataloaders(train_dataset,
                                                               batch_size,
                                                               validation_size=validation_size,
                                                               seed=seed)
        return train_loader, valid_loader

def get_dataloader(dataset,
                   batch_size,
                   shuffle=False,
                   num_workers=0,
                   pin_memory=False):
    """ 
    A straightforward call to torch.utils.data.DataLoader 
    
    Args:
        - shuffle: bool, whether data order is shuffled (each epoch). 
        - pin memory: DataLoader allocate the samples in page-locked memory, which speeds-up the transfer
                      Set true if dataset is on CPU, False if data is already pushed to the GPU. 
        - num_workers: if 0 main process does the dataloading. 
    """
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size,
                                             sampler=sampler,
                                             num_workers=num_workers, 
                                             pin_memory=pin_memory)
    
    return dataloader

def get_mnist_datasets(flatten_bool=True):
    network_mnist_folder = "/network/datasets/mnist.var/mnist_torchvision"
    mnist_folder   = utils.copy_folder_to_slurm_tmpdir(network_mnist_folder)
    # dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # full_trainset = torchvision.datasets.MNIST(root=mnist_folder, train=True,
    #                                            download=False, transform=dataset_transform)

    # test_set  = torchvision.datasets.MNIST(root=mnist_folder, train=False,

    full_trainset = MNISTDataset(mnist_folder/"MNIST/processed/training.pt", flatten_bool)
    test_set = MNISTDataset(mnist_folder/"MNIST/processed/test.pt", flatten_bool)
    
    return full_trainset, test_set

def get_fashion_mnist_datasets(flatten_bool=True):
    fashion_mnist_path = get_torchvision_data("FashionMNIST")
    fashion_mnist_path_slurm = utils.copy_folder_to_slurm_tmpdir(fashion_mnist_path)
    
    full_trainset = MNISTDataset(fashion_mnist_path_slurm/"processed/training.pt", flatten_bool)
    test_set = MNISTDataset(fashion_mnist_path_slurm/"processed/test.pt", flatten_bool)
    
    return full_trainset, test_set

def get_mnist_train_test_dataloaders(batch_size, test_batch_size=None, shuffle=True,
                                     num_workers=0, pin_memory=False,):

    """
    A simple wrapper around torch.utils.data.DataLoader

    Args:
        - shuffle: bool, whether data is shuffled (each epoch).
        - pin memory: DataLoader allocate the samples in page-locked memory, which speeds-up the transfer
                      Set true if dataset is on CPU, False if data is already pushed to the GPU.
        - num_workers: if 0 main process does the dataloading.
    """
    # todo pass in a string for the dataset?
    full_trainset, test_set = get_mnist_datasets()

    if test_batch_size is None:
        test_batch_size = batch_size

    train_loader = torch.utils.data.DataLoader(full_trainset, batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(test_set, test_batch_size,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return train_loader, test_loader

# export
def get_mnist_train_eval_dataloaders(batch_size, val_size=10000, val_batch_size=None,
                                     num_workers=0, pin_memory=False,):
    """
    # todo - pass in a string for the dataset?

    Returns:
         train dataloader, val dataloader
    """
    full_trainset, _ = get_mnist_datasets()
    return get_train_val_dataloaders(full_trainset, val_size, batch_size, val_batch_size,
                                     num_workers, pin_memory)


def get_train_val_dataloaders(dataset, val_size, batch_size, val_batch_size=None,
                              num_workers=0, pin_memory=False,):
    """
    Splits a dataset into a training and validation sets and returns dataloaders.

    Creates two SubsetRandomSamplers (see https://pytorch.org/docs/1.1.0/data.html)
    Therefore batches will always be shuffled.

    Args:
        - seed: int or None, if not None the data indices used for the validation set are shuffled.
        - pin memory: DataLoader allocate the samples in page-locked memory, which speeds-up the transfer
                      Set true if dataset is on CPU, False if data is already pushed to the GPU.
        - num_workers: if 0 main process does the dataloading.
    """
    assert val_size > 0
    if val_batch_size is None:
        val_batch_size = batch_size

    dataset_size = len(dataset)
    data_indices = list(range(dataset_size))
    np.random.shuffle(data_indices) # in-place, seed set outside function

    train_idx = data_indices[val_size:]
    valid_idx = data_indices[:val_size]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler   = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(dataset,
                                             val_batch_size,
                                             sampler=val_sampler,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)

    return train_loader, val_loader

def get_train_eval_val_dataloaders(training_dataset, dataset, val_size, batch_size, val_batch_size=None,
                                   num_workers=0, pin_memory=False):
    """
    training_dataset : dataset with augmentations
    dataset : same as training_dataset but without augmentations
    """
    assert val_size > 0
    if val_batch_size is None:
        val_batch_size = batch_size
    assert len(training_dataset) == len(dataset)

    dataset_size = len(dataset)
    data_indices = list(range(dataset_size))
    np.random.shuffle(data_indices) # in-place, seed set outside function

    train_idx = data_indices[val_size:]
    valid_idx = data_indices[:val_size]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_eval_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler   = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(training_dataset,
                                               batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    train_eval_loader = torch.utils.data.DataLoader(dataset,
                                               val_batch_size,
                                               sampler=train_eval_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(dataset,
                                             val_batch_size,
                                             sampler=val_sampler,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)

    return train_loader, train_eval_loader, val_loader