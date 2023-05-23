# export
import os
import shutil
import collections

import numpy as np
import torch

from pathlib import Path

from lib import utils

# from lib import nbdevlite

# export
import urllib.request

# following code is based heavily on:
# https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py

def download_ptb():
    "Downloads penn treebank dataset to directory, skips if files exist"

    urls=['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
          'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
          'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']

    directory = "/network/projects/_groups/linclab_users/danns/playground/data/ptb"
    data_dir = Path(directory)
    for url in urls:
        filepath = data_dir/Path(url).name
        if filepath.is_file():
            pass
            #print(f'{filepath} already exists, skipping download')
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f'Downloading {filepath.name}')
            urllib.request.urlretrieve(url, filename=filepath)
    return data_dir

# export
def read_words(filename) -> list:
    """Returns list of words with \n replaced with <eos> """
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def build_vocab(filename):
    """builds word to id and id to word dictonaries"""
    data = read_words(filename)

    word_count_tuples = collections.Counter(data).items()
    word_count_tuples = sorted(word_count_tuples, key=lambda x: (-x[1], x[0]))

    # Use the word frequency ordering as id
    word_to_id = {x[0]:i for i, x in enumerate(word_count_tuples)}
    id_to_word = {i:x[0] for i, x in enumerate(word_count_tuples)}

    return word_to_id, id_to_word

# export
def file_to_word_ids(filename, word_to_id:dict)->list:
    """Converts text to a list of word_ids skipping words if not in word_to_id"""
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_ptb_data(ptb_dir):
    """Returns PTB data as lists of word_ids and word<->id dicts"""
    ptb_dir = Path(ptb_dir)
    word_to_id, id_2_word = build_vocab(ptb_dir/"ptb.train.txt")
    # converts valid and test using training set word_to_id dict
    train_ids = file_to_word_ids(ptb_dir/"ptb.train.txt", word_to_id)
    valid_ids = file_to_word_ids(ptb_dir/"ptb.test.txt", word_to_id)
    test_ids  = file_to_word_ids(ptb_dir/"ptb.test.txt", word_to_id)
    return train_ids, valid_ids, test_ids, word_to_id, id_2_word

# export
def ptb_iterator(word_ids:list, batch_size:int, num_steps:int, to_device=False):
    """Iterate on PTB word-ids.

      This generates batch_size pointers into the raw PTB data, and allows
      minibatch iteration along these pointers.
      Args:
        word_ids:   list of word_ids
        batch_size: the batch size.
        num_steps:  the number of unrolls/timesteps.
        to_device:  puts the word_id_arr to
      Yields:
        Pairs (tuples) of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.

      Raises ValueError if batch_size or num_steps are too high.

    # Todo: pytorch expects seq len as first dim.
    """
    data_len = len(word_ids)
    total_steps_per_epoch = data_len // batch_size

    i = batch_size * total_steps_per_epoch # a slice index as we discard some ids
    word_id_arr = np.array(word_ids, dtype=np.int64)[:i] # pytorch embedding requires LongTensor
    word_id_arr = word_id_arr.reshape(batch_size, total_steps_per_epoch)
    word_id_arr = torch.from_numpy(word_id_arr)
    if to_device:
        word_id_arr = word_id_arr.to(utils.get_device())

    n_batches_per_epoch = (total_steps_per_epoch - 1) // num_steps # -1 as y = x_t+1

    if n_batches_per_epoch == 0:
        raise ValueError("n_batches_per_epoch == 0, decrease batch_size or num_steps")

    for i in range(n_batches_per_epoch):
        x = word_id_arr[:, i*num_steps:(i+1)*num_steps]
        y = word_id_arr[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

# export
def load_ptb():
    """
    Fill out documentation
    """
    data_dir = download_ptb()
    ptb_slurm_path = utils.copy_folder_to_slurm_tmpdir(data_dir)
    train_ids, valid_ids, test_ids, word_to_id, id_2_word = load_ptb_data(ptb_slurm_path)
    return train_ids, valid_ids, test_ids, word_to_id, id_2_word

# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    # 
    pass
