from torchnlp.datasets import penn_treebank_dataset
from torchnlp.word_to_vector import GloVe
import torch
import os
from collections import Counter
import hashlib
from lib import utils
from lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from lib.base_rnn import RNNCell 
from lib.model import Model
from lib.dann_rnn import EiRNNCell
from lib.update_policies import EiRNN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, DalesANN_cSGD_UpdatePolicy, ColumnEiDenseSGD, SGD_Clip_UpdatePolicy
from lib.init_policies import W_HybridInit_Uniform, ColumnEi_FirstCell_U_InitPolicy, ColumnEiCell_W_InitPolicy, W_TorchInit, U_TorchInit
from lib.song_rnn import ColumnEiCell
import torch.nn as nn
import torch.nn.functional as F
from config import PLAYGROUND_DIR
from argparse import ArgumentParser
import itertools
import imp
from pprint import pprint
import numpy as np
import math


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, cuda=True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # data = data.view(bsz, -1).contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, seq_len=None, evaluation=False, bptt=20):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len].view(-1)
    target = source[i+1:i+1+seq_len]
    return data, target


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    
parser = ArgumentParser()
parser.add_argument('--array', type=int)
parser.add_argument('--model_type', type=str)
args = parser.parse_args()
array_index = args.array
model_type = args.model_type # song / danns / rnn

results_dir = PLAYGROUND_DIR/f"ptb_3layers_{model_type}_icml_test"
    
fn = '/network/projects/linclab_users/danns/playground/corpus.{}.data'.format(hashlib.md5('/network/projects/linclab_users/danns/playground/data/ptb'.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = Corpus('/network/projects/linclab_users/danns/playground/data/ptb')
    torch.save(corpus, fn)
    
batch_size = 64
time_steps = 50
val_batch_size = len(corpus.valid) // (time_steps+1)
test_batch_size= len(corpus.test) // (time_steps+1)
emsize = 300
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, val_batch_size)
test_data = batchify(corpus.test, test_batch_size)
vecs = GloVe(name='6B', dim=emsize)

input_features = emsize
n_output_classes  = len(corpus.dictionary)
ntokens = len(corpus.dictionary)
n_hidden = 500
n_epochs = 3
criterion = None
device = 'cuda'

# grad_clip_list = [1, 5, 10, 50, None]
# lr_list = np.geomspace(start=3., stop=3e-2, num=8, endpoint=True)
# seeds = [0,1,2,3,4]
# para_comb = list(itertools.product(seeds, lr_list, grad_clip_list)) 
# (seed, lr, max_gn) = para_comb[array_index-1]
seed = array_index-1

val_interval = 50
decay = .99
threshold = 20.
# model_type = "ei"
if model_type == "song":
    lr = 1.554
    max_gn = 1
    init_lr = round(lr, 3)
    cells = [ColumnEiCell(input_features, (n_hidden, 9),nonlinearity=F.relu, i2h_init_policy=ColumnEi_FirstCell_U_InitPolicy(dataset='PennTreebank'), h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=1.5), update_policy=ColumnEiSGD_Clip(max=max_gn)),
             ColumnEiCell(n_hidden, (n_hidden, 9),nonlinearity=F.relu, h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=1.5), update_policy=ColumnEiSGD_Clip(max=max_gn)),
             ColumnEiCell(n_hidden, (n_hidden, 9),nonlinearity=F.relu, h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=1.5), update_policy=ColumnEiSGD_Clip(max=max_gn)),
             ColumnEiDense(n_hidden, (n_hidden, 9), nonlinearity=F.relu, update_policy=ColumnEiDenseSGD(max=max_gn)),
             ColumnEiDense(n_hidden, (n_hidden, 9), nonlinearity=F.relu, update_policy=ColumnEiDenseSGD(max=max_gn)),
             ColumnEiDense(n_hidden, (n_output_classes, 9), nonlinearity=None, update_policy=ColumnEiDenseSGD(max=max_gn))]
elif model_type == "danns":
    lr = 0.805
    max_gn = 1
    init_lr = round(lr, 3)
    cells = [EiRNNCell(input_features, n_hidden, n_hidden//10, n_hidden//10,nonlinearity=F.relu),
             EiRNNCell(n_hidden, n_hidden, n_hidden//10, n_hidden//10,nonlinearity=F.relu),
             EiRNNCell(n_hidden, n_hidden, n_hidden//10, n_hidden//10,nonlinearity=F.relu),
             EiDense(n_hidden, n_hidden, n_hidden//10, nonlinearity=F.relu),
             EiDense(n_hidden, n_hidden, n_hidden//10, nonlinearity=F.relu),
             EiDense(n_hidden, n_output_classes, n_output_classes//10, nonlinearity=None)] 
    for cell in cells[-3:]:
        cell.update_policy = DalesANN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
    for cell in cells[:-3]:
        cell.update_policy = EiRNN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
else:
    lr = 0.805
    max_gn = 1
    init_lr = round(lr, 3)
    cells =[RNNCell(input_features, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_TorchInit(), update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)),
            RNNCell(n_hidden, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_TorchInit(), update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)),
            RNNCell(n_hidden, n_hidden, nonlinearity=F.relu, i2h_init_policy=U_TorchInit(), h2h_init_policy=W_TorchInit(), update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)),
            DenseLayer(n_hidden, n_hidden, nonlinearity=F.relu, update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)),
            DenseLayer(n_hidden, n_hidden, nonlinearity=F.relu, update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn)),
            DenseLayer(n_hidden, n_output_classes, nonlinearity=None, update_policy=SGD_Clip_UpdatePolicy(max_grad_norm=max_gn))]


def train(lr, n_epochs, cells, seed, max_gn=None, decay=decay, model_type=model_type):
    
    params = utils.Params()
    params.model_type = model_type
    params.max_gn = max_gn
    params.seed = seed
    params.batch_size = 64
    params.n_epochs = n_epochs
    params.lr  = lr
    params.n_hidden = 500
    params.results_dir = results_dir
    val_interval = 50
    n_batches_to_average_for_train_performance = 10

    device = utils.get_device()
    utils.set_seed_all(params.seed)
    
    model = Model(cells)
    model.init_weights()
    model.reset_hidden(batch_size=params.batch_size)
    model.to(device)

    params.write_to_config_yaml()

    buffer_size = n_batches_to_average_for_train_performance
    train_ppl_buffer = np.empty(buffer_size)
    train_ppl_buffer[:] = np.nan # slice assignment is inplace
    train_loss_buffer = np.copy(train_ppl_buffer)
    update_i = 0   # update counter
    results = {k:[] for k in ["test_loss", "test_ppl", "val_loss", "val_ppl", "train_loss", "train_ppl", "update"]}


    for epoch in range(n_epochs):
        # if epoch % 2 == 0 and epoch != 0 : lr = lr * 0.98
        for iter_id, batch_ind in enumerate(np.random.permutation(np.arange(train_data.size(0) - 1 - 1))):
            model.train()
            model.reset_hidden(batch_size=batch_size)
            data, targets = get_batch(train_data, batch_ind, seq_len=time_steps)
            input_list = []
            for t in range(data.size(0)):
                words_list = []
                for idx in data[t].tolist():
                    words_list.append(corpus.dictionary.idx2word[idx])
                word_embeddings = vecs[words_list]
                input_list.append(word_embeddings)
                # print(word_embeddings.shape)
            data = torch.stack(input_list)
            targets.to(device)
            output_list = []
            for t in range(data.size(0)):
                output = model(data[t].to(device))
                output_list.append(output)
            logits = torch.stack(output_list).reshape(-1, n_output_classes)
            targets = targets.reshape(-1)
            out = F.cross_entropy(logits, targets)
            out.backward()
            model.update(lr=lr, max_grad_norm=max_gn)
            model.zero_grad()
            update_i += 1 # update counter

            idx = iter_id % buffer_size 
            if out.item() <= threshold: train_ppl_buffer[idx]  = np.e**(out.item()) 
            else: train_ppl_buffer[idx] = np.nan
            train_loss_buffer[idx] = out.item()

            if iter_id % val_interval == 0:
                lr = lr * decay
                if out.item() <= threshold: print(f'e-{epoch}-iter-{iter_id}: train loss:{out}; train perplexity:{np.e**(out)}')
                else: print(f'e-{epoch}-iter-{iter_id}: train loss:{out}; train perplexity: nan')
                model.eval()
                model.reset_hidden(batch_size=val_batch_size)
                data, targets = get_batch(val_data, 0, seq_len=time_steps)
                input_list = []
                for t in range(data.size(0)):
                    words_list = []
                    for idx in data[t].tolist():
                        words_list.append(corpus.dictionary.idx2word[idx])
                    word_embeddings = vecs[words_list]
                    input_list.append(word_embeddings)
                    # print(word_embeddings.shape)
                data = torch.stack(input_list)
                targets.to(device)
                output_list = []
                for t in range(data.size(0)):
                    output = model(data[t].to(device))
                    output_list.append(output)
                logits = torch.stack(output_list).reshape(-1, n_output_classes)
                targets = targets.reshape(-1)            
                out = F.cross_entropy(logits, targets)


                if out.item() <= threshold: 
                    print(f'e-{epoch}-iter-{iter_id}: val loss:{out}; val perplexity:{np.e**(out.item())}')
                    results["update"].append(update_i)
                    results["train_ppl"].append(np.mean(train_ppl_buffer))
                    results["train_loss"].append(np.mean(train_loss_buffer))
                    results["val_ppl"].append(np.e**(out.item()))
                    results["val_loss"].append(out.item())
    
                else: 
                    print(f'e-{epoch}-iter-{iter_id}: val loss:{out}; val perplexity: nan')
                    results["update"].append(update_i)
                    results["train_ppl"].append(np.mean(train_ppl_buffer))
                    results["train_loss"].append(np.mean(train_loss_buffer))
                    results["val_ppl"].append(np.nan)
                    results["val_loss"].append(out.item())
                    
                model.reset_hidden(batch_size=test_batch_size)
                data, targets = get_batch(test_data, 0, seq_len=time_steps)
                input_list = []
                for t in range(data.size(0)):
                    words_list = []
                    for idx in data[t].tolist():
                        words_list.append(corpus.dictionary.idx2word[idx])
                    word_embeddings = vecs[words_list]
                    input_list.append(word_embeddings)
                    # print(word_embeddings.shape)
                data = torch.stack(input_list)
                targets.to(device)
                output_list = []
                for t in range(data.size(0)):
                    output = model(data[t].to(device))
                    output_list.append(output)
                logits = torch.stack(output_list).reshape(-1, n_output_classes)
                targets = targets.reshape(-1)            
                out = F.cross_entropy(logits, targets)


                if out.item() <= threshold: 
                    print(f'e-{epoch}-iter-{iter_id}: test loss:{out}; test perplexity:{np.e**(out.item())}')
                    results["test_ppl"].append(np.e**(out.item()))
                    results["test_loss"].append(out.item())
                else: 
                    print(f'e-{epoch}-iter-{iter_id}: test loss:{out}; test perplexity: nan')
                    results["test_ppl"].append(np.nan)
                    results["test_loss"].append(out.item())
                print('-'*50)

    params.get_exp_savepath().mkdir(parents=True, exist_ok=True)
    filestr = str(params.get_exp_savepath() / f'{model_type}_icml_learning_curves_seed{seed}_lr{params.lr}_GC{max_gn}')
    np.savez(filestr, **results)
    return results, model

results, model = train(lr, n_epochs, cells, seed, max_gn=max_gn, decay=decay, model_type=model_type)
torch.save(model.state_dict(), f'{PLAYGROUND_DIR}/saved_models/ptb_3layers_test_{model_type}_seed{seed}_lr{init_lr}_GC{max_gn}_icml.pth')
