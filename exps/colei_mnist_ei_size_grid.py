import imp
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from argparse import ArgumentParser
import itertools

from lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from lib.base_rnn import RNNCell 
from lib.model import Model
from lib.dann_rnn import EiRNNCell
from lib.song_rnn import ColumnEiCell
from lib.update_policies import EiRNN_cSGD_UpdatePolicy, DalesANN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, ColumnEiDenseSGD
from lib.init_policies import EiDenseWeightInit_WexMean, U_TorchInit, W_TorchInit, ColumnEiCell_W_InitPolicy, ColumnEi_FirstCell_U_InitPolicy, EiRNNCell_U_InitPolicy, EiRNNCell_W_InitPolicy, W_UniformInit

from lib import utils
from lib import rnn_basic_tasks
from lib.utils import acc_func
from config import PLAYGROUND_DIR

parser = ArgumentParser()
parser.add_argument('--array', type=int)
parser.add_argument('--model_type', type=str)
args = parser.parse_args()
array_index = args.array
model_type = args.model_type # song / danns / rnn

results_dir = PLAYGROUND_DIR/f"seq_mnist_colei_ei_size_grid"
# We will do row-sequential mnist so 28 rows, 28 inputs
input_features = 28
n_rows = 28
# and only 10 classes
n_output_classes  = 10
n_epochs = 30

seeds = [0,1,2,3,4]
ei_list = (1/np.linspace(0.01,0.5,10)) -1 # 1%-50% linear space
n_hidden_list = np.linspace(100, 1000, 10, dtype=int)

para_comb = list(itertools.product(seeds, ei_list, n_hidden_list))

(seed, ei_ratio, n_hidden) = para_comb[array_index-1]

(lr, max_gn) = (0.01, 5)
rad = 1.5


if model_type == "song":
    
    cells = [ColumnEiCell(input_features, (n_hidden, ei_ratio),nonlinearity=F.relu, i2h_init_policy=ColumnEi_FirstCell_U_InitPolicy(dataset='MNIST'), h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=rad), update_policy=ColumnEiSGD_Clip(max=max_gn)),
            ColumnEiDense(n_hidden, (n_output_classes, ei_ratio), nonlinearity=None, update_policy=ColumnEiDenseSGD(max=max_gn))
            ]
    
params = utils.Params()
params.model_type = model_type
params.max_gn = max_gn
params.seed = seed
params.batch_size = 32
params.test_size = 10000
# params.test_batch_size = 1000
params.n_epochs = n_epochs
params.lr  = lr
params.n_hidden = n_hidden
params.results_dir = results_dir
test_interval = 100
n_batches_to_average_for_train_performance = 10

device = utils.get_device()
utils.set_seed_all(params.seed)

# train_dataloader, val_dataloader = rnn_basic_tasks.get_mnist_train_eval_dataloaders(params.batch_size,
#                                                                                     params.val_size,
#                                                                                     params.val_batch_size)
train_dataloader, test_dataloader = rnn_basic_tasks.get_mnist_train_test_dataloaders(params.batch_size,
                                                                                    test_batch_size=params.test_size)

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

        if update_i % test_interval == 0:
            model.eval()
            test_err_buffer  = []
            test_loss_buffer = []
            for test_batch_i, (x, y) in enumerate(test_dataloader): 
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
                test_err_buffer.append((1 - acc)*100)
                test_loss_buffer.append(loss.item())

            results["update"].append(update_i)
            results["train_err"].append(np.mean(train_err_buffer))
            results["train_loss"].append(np.mean(train_loss_buffer))
            results["test_err"].append(np.mean(test_err_buffer))
            results["test_loss"].append(np.mean(test_loss_buffer))

            # Epoch {epoch_i}, batch {batch_i+1}):\n\
            s = "\r " + f"e{epoch_i}-b{batch_i}-u{update_i}: \
Train err {np.mean(train_err_buffer):.3f} % loss {np.mean(train_loss_buffer):.3f}, \
Test err {np.mean(test_err_buffer):.3f} % loss {np.mean(test_loss_buffer):.3f}"
            print(s)
params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
filestr = str(params.get_exp_savepath() / f'colei_learning_curves_seed{seed}_hidden{n_hidden}_ei{round(ei_ratio,3)}_lr{lr}_GC{max_gn}_rad{rad}')
np.savez(filestr, **results)
torch.save(model.state_dict(), f'{PLAYGROUND_DIR}/saved_models/seq_mnist_shallow_test_{model_type}_model_seed{seed}_hidden{n_hidden}_ei{round(ei_ratio,4)}_lr{lr}_GC{max_gn}_rad{rad}.pth')
