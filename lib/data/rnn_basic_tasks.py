# export

"""
A copy of the basic tasks in the directory above this. 

"""
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from lib import utils

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
