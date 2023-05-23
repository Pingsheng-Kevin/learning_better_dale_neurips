import numpy as np
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import Counter
import hashlib
from lib import utils
from lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from lib.base_rnn import RNNCell 
from lib.model import Model
from lib.dann_rnn import EiRNNCell
from lib.update_policies import EiRNN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, DalesANN_cSGD_UpdatePolicy, ColumnEiDenseSGD
from lib.init_policies import EiDenseWeightInit_WexMean, EiRNNCell_W_InitPolicy, W_HybridInit_Uniform, EiRNNCell_WeightInitPolicy, ColumnEi_FirstCell_U_InitPolicy, ColumnEiCell_W_InitPolicy, W_TorchInit, U_TorchInit, W_UniformInit, EiRNNCell_W_HybridInit
from lib.song_rnn import ColumnEiCell
from config import PLAYGROUND_DIR
from argparse import ArgumentParser
import itertools
import imp
from pprint import pprint

parser = ArgumentParser()
parser.add_argument('--array', type=int)
parser.add_argument('--model_type', type=str)
args = parser.parse_args()
array_index = args.array
model_type = args.model_type # song / danns / rnn

results_dir = PLAYGROUND_DIR/f"adding_10units_{model_type}_icml_test"

adding_data = np.load('/network/projects/linclab_users/RNN_adding_data/adding_data_20.npz')

input_features = 2
n_output_classes = 1
n_hidden = 10
device = 'cuda'
test_interval = 20
n_epochs = 100
test_batch_size = 1000
batch_size = 64
seed = array_index-1
# grad_clip_list = [1, 3, 5, 7, None]
# lr_list = [.5, 1e-1, 5e-2]
# seeds = [0,1,2,3,4,5,6,7,8,9]
# para_comb = list(itertools.product(seeds, lr_list, grad_clip_list)) # 
# (seed, lr, max_gn) = para_comb[array_index-1]

# model_type = 'rnn'
if model_type == "song":
    (lr, max_gn) = (0.1, 7)
    cells = [ColumnEiCell(input_features, (n_hidden, 9),nonlinearity=F.relu, i2h_init_policy=ColumnEi_FirstCell_U_InitPolicy(dataset='Adding'), h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=1.5), update_policy=ColumnEiSGD_Clip(max=max_gn)),
             ColumnEiDense(n_hidden, (n_output_classes, 9), nonlinearity=None, update_policy=ColumnEiDenseSGD(max=max_gn))
            ]
elif "danns" == model_type:
    (lr, max_gn) = (0.1, 3)
    cells = [EiRNNCell(input_features, n_hidden,max(n_hidden//10,1),max(n_hidden//10,1),nonlinearity=F.relu, h2h_init_policy=EiRNNCell_W_InitPolicy(numerator=1/3, random=False)),
            EiDense(n_hidden, n_output_classes, max(n_output_classes//10,1), nonlinearity=None, weight_init_policy=EiDenseWeightInit_WexMean(numerator=2))
            ] 
    for cell in cells[-1:]:
        cell.update_policy = DalesANN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
    for cell in cells[:-1]:
        cell.update_policy = EiRNN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
elif "rnn" == model_type:
    (lr, max_gn) = (0.1, 7)
    cells =[RNNCell(input_features, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_TorchInit()),
            DenseLayer(n_hidden, n_output_classes, nonlinearity=None)]
    
params = utils.Params()
params.model_type = model_type
params.max_gn = max_gn
params.seed = seed
params.batch_size = 64
params.n_epochs = n_epochs
params.lr  = lr
params.n_hidden = 10
params.results_dir = results_dir
n_batches_to_average_for_train_performance = 10

device = utils.get_device()
utils.set_seed_all(params.seed)

params.write_to_config_yaml()

buffer_size = n_batches_to_average_for_train_performance
train_loss_buffer = np.empty(buffer_size)
train_loss_buffer[:] = np.nan # slice assignment is inplace
update_i = 0   # update counter
results = {k:[] for k in ["test_loss", "train_loss", "update"]}
    
model = Model(cells)
model.init_weights()
model.reset_hidden(batch_size=batch_size)
model.to(device)

data = np.expand_dims(adding_data['data_series'], axis=1)
masks = np.expand_dims(adding_data['mask_series'], axis=1)
data_masks = np.concatenate((data,masks), axis=1)
data_train = data_masks[:-1000,:,:]
targets_train = adding_data['res_series'][:-1000]
data_test = data_masks[-1000:,:,:]
targets_test = adding_data['res_series'][-1000:]

decay = 0.99
num_batches = (data_train.shape[0]//batch_size)+1

data_test = torch.from_numpy(data_test).float().to(device)
targets_test = torch.from_numpy(targets_test).float().to(device)

for epoch in range(n_epochs):
    # if epoch % 2 == 0 and epoch != 0 : lr = lr * 0.98
    indices = np.random.permutation(np.arange(data_train.shape[0]))
    for i in range(num_batches):
        if i != num_batches-1: 
            data = data_train[indices[i*batch_size:(i+1)*batch_size]]
            targets = targets_train[indices[i*batch_size:(i+1)*batch_size]]
        else: 
            data = data_train[indices[i*batch_size:]]
            targets = targets_train[indices[i*batch_size:]]
        data = torch.from_numpy(data).float().to(device)
        targets = torch.from_numpy(targets).float().to(device)
        model.train()
        model.reset_hidden(batch_size=data.size(0))
        for t in range(data.shape[2]):
            output = model(data[:,:,t])
        output.squeeze_()
        loss = F.mse_loss(output, targets)
        loss.backward()
        model.update(lr=lr)
        model.zero_grad()
        
        update_i += 1 # update counter

        idx = i % buffer_size 
        train_loss_buffer[idx] = loss.item()
        
        if i % test_interval == 0:
            lr = lr * decay
            print(f'epoch: {epoch} train loss:{np.mean(train_loss_buffer)}')
            model.eval()
            model.reset_hidden(batch_size=test_batch_size)
            for t in range(data_test.shape[2]):
                output = model(data_test[:,:,t])
            output.squeeze_()            
            loss = F.mse_loss(output, targets_test)
            print(f'epoch: {epoch} test loss:{loss}')
            print('-'*30)
            
            results["update"].append(update_i)
            results["train_loss"].append(np.mean(train_loss_buffer))
            results["test_loss"].append(loss.item())
            
params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
filestr = str(params.get_exp_savepath() / f'{model_type}_icml_learning_curves_seed{seed}_lr{params.lr}_GC{max_gn}_decay{decay}')
np.savez(filestr, **results)
torch.save(model.state_dict(), f'{PLAYGROUND_DIR}/saved_models/adding_{model_type}_seed{seed}_lr{params.lr}_GC{max_gn}_decay{decay}_icml.pth')
