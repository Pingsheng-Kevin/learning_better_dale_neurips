
import imp
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from argparse import ArgumentParser
import itertools

from lib.dense_layers import DenseLayer, EiDense
from lib.base_rnn import RNNCell 
from lib.model import Model
from lib.dann_rnn import EiRNNCell
from lib.update_policies import EiRNN_cSGD_UpdatePolicy, DalesANN_cSGD_UpdatePolicy, SGD_Clip_UpdatePolicy
from lib.init_policies import EiRNNCell_W_InitPolicy, W_UniformInit, U_TorchInit

from lib import utils
from lib import rnn_basic_tasks
from lib.utils import acc_func
from config import PLAYGROUND_DIR

results_dir = PLAYGROUND_DIR/"seq_mnist_rnn_neural_comp"

def train(lr, n_epochs, cells, seed=0, max_gn=None, results_dir=results_dir, rad=None):

    params = utils.Params()
    params.model_type = "ei"
    params.max_gn = max_gn
    params.seed = seed
    params.batch_size = 32
    params.val_size = 10000
    params.val_batch_size = 1000
    params.n_epochs = n_epochs
    params.lr  = lr
    params.n_hidden = 100
    params.results_dir = results_dir
    val_interval = 100
    n_batches_to_average_for_train_performance = 10

    device = utils.get_device()
    utils.set_seed_all(params.seed)

    train_dataloader, val_dataloader = rnn_basic_tasks.get_mnist_train_eval_dataloaders(params.batch_size,
                                                                                        params.val_size,
                                                                                        params.val_batch_size)
    
    # x, y = next(iter(train_dataloader))
    # x    = utils.batch_last(x, flatten=False, to_device=False)
    

    n_rows = 28

    model = Model(cells)
    model.init_weights()
    model.reset_hidden(batch_size=params.batch_size)
    model.to(device)

    params.write_to_config_yaml()

    buffer_size = n_batches_to_average_for_train_performance
    train_err_buffer = np.empty(buffer_size)
    train_err_buffer[:] = np.nan # slice assignment is inplace
    train_loss_buffer = np.copy(train_err_buffer)


    results = {k:[] for k in ["test_loss", "test_err", "train_loss", "train_err", "update"]}
    update_i = 0   # update counter
    for epoch_i in range(params.n_epochs):
        for batch_i, (x, y) in enumerate(train_dataloader):
            y = y.to(device)
            x = x.to(device) # x is of shape B=bs x C=1 x H=28, W=28 
            
            model.train()
            model.reset_hidden(x.shape[0])
            for row_i in range(n_rows):
                # x_row = utils.batch_last(x[:, 0, row_i, :])
                x_row = x[:, 0, row_i, :]
                yhat  = model(x_row)
                
            # loss = F.cross_entropy(yhat.T,y)
            loss = F.cross_entropy(yhat,y)
            loss.backward()
            model.update(lr=params.lr, max_grad_norm=max_gn)
            model.zero_grad()
            update_i += 1
            
            # acc = acc_func(yhat.T, y)
            acc = acc_func(yhat, y)
            err = (1 - acc)*100
            
            idx = batch_i % buffer_size 
            #print(idx, buffer_size)
            train_err_buffer[idx]  = err
            train_loss_buffer[idx] = loss.item()
            
            if update_i % val_interval == 0:
                model.eval()
                val_err_buffer  = []
                val_loss_buffer = []
                for val_batch_i, (x, y) in enumerate(val_dataloader): 
                    y = y.to(device)
                    x = x.to(device) # of shape B=bs x C=1 x H=28, W=28 ]
                    model.reset_hidden(x.shape[0])
                    for row_i in range(n_rows):
                        # x_row = utils.batch_last(x[:, 0, row_i, :])
                        x_row = x[:, 0, row_i, :]
                        yhat  = model(x_row)
                    # loss = F.cross_entropy(yhat.T,y)
                    loss = F.cross_entropy(yhat,y)
                    # acc = acc_func(yhat.T, y)
                    acc = acc_func(yhat, y)
                    val_err_buffer.append((1 - acc)*100)
                    val_loss_buffer.append(loss.item())
                
                results["update"].append(update_i)
                results["train_err"].append(np.mean(train_err_buffer))
                results["train_loss"].append(np.mean(train_loss_buffer))
                results["test_err"].append(np.mean(val_err_buffer))
                results["test_loss"].append(np.mean(val_loss_buffer))
                    
                # Epoch {epoch_i}, batch {batch_i+1}):\n\
                s = "\r " + f"e{epoch_i}-b{batch_i}-u{update_i}: \
    Train err {np.mean(train_err_buffer):.3f} % loss {np.mean(train_loss_buffer):.3f}, \
    Test err {np.mean(val_err_buffer):.3f} % loss {np.mean(val_loss_buffer):.3f}    "
                print(s)
    params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
    filestr = str(params.get_exp_savepath() / f'rnn_learning_curves_seed{seed}_num_rec_cells{num_rec_cells}_hidden{n_hidden}_lr{lr}_GC{max_gn}_rad{rad}')
    np.savez(filestr, **results)
    return results, model
    # params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
    # filestr = str(params.get_exp_savepath() / 'learning_curves')
    # np.savez(filestr, **results)
    # to load use np.load(filestr+'.npz')["train_err"]

parser = ArgumentParser()
parser.add_argument('--array', type=int)
args = parser.parse_args()
array_index = args.array
# We will do row-sequential mnist so 28 rows, 28 inputs
input_features = 28
n_rows = 28
# and only 10 classes
n_output_classes  = 10
# n_hidden = 100 
n_epochs = 30
# max_gn = 1e8
model_type = "rnn"

# DANNs with Torch Spectrum
grad_clip_list = [1, 5, 10, None]
lr_list = np.geomspace(5e-2, 5e-4, num=7, endpoint=True)
# rad_list = [2+i for i in range(7)]
rad_list = [0.4+0.1*i for i in range(7)]
seeds = [i for i in range(5)]
# num_rec_cells_list = [1, 2, 3]
# n_hidden_list = [50, 100, 200]
para_comb = list(itertools.product(seeds, lr_list, grad_clip_list, rad_list)) # 
(num_rec_cells, n_hidden) = (1, 100)

(seed, lr, max_gn, rad) = para_comb[array_index-1]
lr = round(lr, 6)
num = (rad/2)**2


cells =[RNNCell(input_features, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_UniformInit(num=num), update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn))]

for i in range(num_rec_cells-1): 
    cells.append(RNNCell(n_hidden, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_UniformInit(num=num), update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)))
    
cells.append(DenseLayer(n_hidden, n_output_classes, nonlinearity=None, update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)))


results, model = train(lr, n_epochs, cells, seed, max_gn=max_gn, rad=rad)
torch.save(model.state_dict(), f'{PLAYGROUND_DIR}/saved_models/rnn_model_seed{seed}_num_rec_cells{num_rec_cells}_hidden{n_hidden}_lr{lr}_GC{max_gn}_rad{rad}.pth')
    
"""
params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
filestr = str(params.get_exp_savepath() / 'learning_curves')
np.savez(filestr, **results)
# to load use np.load(filestr+'.npz')["train_err"]
"""
