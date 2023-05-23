import itertools
import imp
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from argparse import ArgumentParser

from lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from lib.base_rnn import RNNCell 
from lib.model import Model
from lib.dann_rnn import EiRNNCell
from lib.song_rnn import ColumnEiCell
from lib.update_policies import EiRNN_cSGD_UpdatePolicy, DalesANN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, ColumnEiDenseSGD, SGD_Clip_UpdatePolicy
from lib.init_policies import EiDenseWeightInit_WexMean, W_UniformInit, U_TorchInit, W_TorchInit, ColumnEiCell_W_InitPolicy, ColumnEi_FirstCell_U_InitPolicy, EiRNNCell_U_InitPolicy, EiRNNCell_W_InitPolicy, ColumnEi_Dense_InitPolicy, W_RNNInit_ColEI_Spectrum_Init, U_ColEIInit, W_ColEIInit_RNN_Spectrum_Init, Dense_RNNInit_ColEI_Spectrum_Init, U_RNNInit_ColEI_Spectrum_Init,Dense_ColEIInit_RNN_Spectrum_Init,W_ColEIInit_RNN_Spectrum_Init,U_ColEIInit_RNN_Spectrum_Init
from lib import utils
from lib import rnn_basic_tasks
from lib.utils import acc_func
from config import PLAYGROUND_DIR

results_dir = PLAYGROUND_DIR/"seq_mnist_sign_rnn_icml_rebuttal"

def train(lr, n_epochs, cells, seed=0, max_gn=None, results_dir=results_dir, rad=None):

    params = utils.Params()
    params.model_type = "rnn"
    params.max_gn = max_gn
    params.seed = seed
    params.batch_size = 32
    params.val_size = 10000
    params.test_size = 10000
    params.val_batch_size = 10000
    params.n_epochs = n_epochs
    params.lr  = lr
    params.n_hidden = 100
    params.results_dir = results_dir

    val_interval = 100
    n_batches_to_average_for_train_performance = 10

    device = utils.get_device()
    utils.set_seed_all(params.seed)

#     train_dataloader, val_dataloader = rnn_basic_tasks.get_mnist_train_eval_dataloaders(params.batch_size,
#                                                                                         params.val_size,
#                                                                                         params.val_batch_size)
    train_dataloader, test_dataloader = rnn_basic_tasks.get_mnist_train_test_dataloaders(params.batch_size,
                                                                                    test_batch_size=params.test_size)
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
            model.update(lr=params.lr)
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
                for val_batch_i, (x, y) in enumerate(test_dataloader): 
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
    filestr = str(params.get_exp_savepath() / f'sign_rnn_learning_curves_seed{seed}_lr{lr}_GC{max_gn}')
    np.savez(filestr, **results)
    return results, model
    # params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
    # filestr = str(params.get_exp_savepath() / 'learning_curves')
    # np.savez(filestr, **results)
    # to load use np.load(filestr+'.npz')["train_err"]

# parser = ArgumentParser()
#parser.add_argument('-a', '--array')
parser = ArgumentParser()
parser.add_argument('--array', type=int)
parser.add_argument('--model_type', type=str)
args = parser.parse_args()
array_index = args.array
model_type = args.model_type # song / danns / rnn

# We will do row-sequential mnist so 28 rows, 28 inputs
input_features = 28
n_rows = 28
# and only 10 classes
n_output_classes  = 10
# n_hidden = 50
n_epochs = 30
# grad_clip_list = [1, 5, 10, None]
# lr_list = np.geomspace(5e-2, 5e-4, num=7, endpoint=True)

# seeds = [i for i in range(5)]
# # num_rec_cells_list = [1, 2, 3]
# # n_hidden_list = [50, 100, 200]
# para_comb = list(itertools.product(seeds, lr_list, grad_clip_list)) # 
(num_rec_cells, n_hidden, rad) = (1, 100, 1.5)

# (seed, lr, max_gn) = para_comb[array_index-1]

# lr = round(lr, 6)
seed = array_index-1
max_gn = None
lr = 0.01
model_type = "rnn"

cells =[RNNCell(input_features, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_TorchInit(), update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)),
        DenseLayer(n_hidden, n_output_classes, nonlinearity=None, update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn))] 

results, model = train(lr, n_epochs, cells, seed, max_gn=max_gn, rad=rad)
torch.save(model.state_dict(), f'{PLAYGROUND_DIR}/saved_models/seq_mnist_rebuttal_sign_rnn_seed{seed}_lr{lr}_GC{max_gn}.pth')


