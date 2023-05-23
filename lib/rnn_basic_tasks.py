# export
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from lib import utils

# export
def get_mnist_datasets():
    network_mnist_folder = "/network/datasets/mnist.var/mnist_torchvision"
    mnist_folder   = utils.copy_folder_to_slurm_tmpdir(network_mnist_folder)
    dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    full_trainset = torchvision.datasets.MNIST(root=mnist_folder, train=True,
                                               download=False, transform=dataset_transform)

    test_set  = torchvision.datasets.MNIST(root=mnist_folder, train=False,
                                          download=False, transform=dataset_transform)

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

#export
def adding_problem_generator(batch_size, seq_len, high=1):
    """ A data generator for adding problem.

    Note: We probably want to generate this once and then store it.

    Original implementation
    # https://minpy.readthedocs.io/en/latest/tutorial/rnn_tutorial/rnn_tutorial.html
    # https://github.com/dmlc/minpy/blob/master/examples/utils/data_utils.py

    Returns a tuple of tensors (X, y)
    - X is a 3 dim tensor of shape (batch_size, seq_len, 2)
        The first column of X[i] is a list of random data; the second
        column is a binary mask, all zeros apart from 2 entries set to 1
    - y is of shape (batch_size,1)
        Each y[i] is the sum of the masked data of X[i], ie. X[1]^TX[2]
    example:

     X.T                y
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    Args:
        batch_size: the number of datapoints to produce.
        seq_len:    the length of a single sequence.
        high:       the random data is sampled from a [0, high] uniform distribution.

    The data follows the adding problem as described in Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units. Not sure if different from the original he adding problem, first proposed
    by Hochreiter and Schmidhuber (1997).
    """
    X_num  = np.random.uniform(low=0, high=high, size=(batch_size, seq_len, 1))
    X_mask = np.zeros((batch_size, seq_len, 1))
    Y = np.ones((batch_size, 1))
    for i in range(batch_size):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)

    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass
