#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-08-03 13:06:43.588260
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import math
import subprocess
import linecache
from tqdm import tqdm

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import torch
from torch.utils.data import Dataset, ConcatDataset, IterableDataset
from dali_dataset import TFRecordPipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np

from projects.feed.rank.src.config import *

# text dataset is slow when file is large (each line with more features like dense embedding 128)
class TextDataset(Dataset):
  def __init__(self, filename, td):
    self._filename = filename
    self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0]) 
    self.td = td 

  def __getitem__(self, idx):
    line = linecache.getline(self._filename, idx + 1)
    # lis, list, list, scalar, scalar
    feat_id, feat_field, feat_value, [label], [id] = self.td.parse_line(line, decode=False)
    ## this will use lele.NpDictPadCollate

    duration = 123
    
    return {'index': feat_id, 'field': feat_field, 'value': feat_value, 'id': id, 'duration': duration}, label
  
    X = {'index': torch.tensor(feat_id), 'field': torch.tensor(feat_field), 'value': torch.tensor(feat_value), 'id': id} 
    
    if FLAGS.use_doc_emb:
      X['doc_emb'] = torch.tensor(doc_emb)
    if FLAGS.use_user_emb:
      X['user_emb'] = torch.tensor(user_emb)

    y = torch.tensor(label)

    return X, y
    
  def __len__(self):
    return self._total_data

# Lmdb is slow, depreciated
class LmdbDataset(Dataset):
  def __init__(self, dir):
    import pyxis.torch as pxt
    self._dataset = pxt.TorchDataset(dir)

  def __getitem__(self, idx):
    data = self._dataset[idx]
    db = self._dataset.db[idx]
    y = data['click']
    del data['click']

    X = data
    y = y.squeeze().float()
    
    if FLAGS.duration_weight:
      X['duration'] = 60 if db['duration'] > 60 * 60 * 12 else db['duration']
      X['weight'] = math.log(X['duration'] + 1.)
      X['weight'] += 1. - db['click']
      if FLAGS.duration_ratio < 1.:
        ratio = FLAGS.duration_ratio
        X['weight'] = X['weight'] * ratio + (1 - ratio)
      X['duration'] = torch.as_tensor(X['duration'])
      X['weight'] = torch.as_tensor(X['weight'])

      X['duration'].squeeze()
      X['weight'].squeeze()

    return X, y
    
  def __len__(self):
    return len(self._dataset)

class TFRecordDataset(IterableDataset):
  def __init__(self, records, batch_size, features, shuffle=False, collate_fn=None, num_workers=1, rank=0):
    super(TFRecordDataset, self).__init__()
    
    self.rank = rank
    self.batch_size = batch_size
    if collate_fn:
      batch_size = 1
    self.batch_size_ = batch_size
    self.collate_fn = collate_fn

    pipes = [TFRecordPipeline(batch_size=batch_size, features=features, path=records, 
                              index_path=[x.replace('tfrecords', 'tfrecords.idx') for x in records], 
                              num_gpus=num_workers, shuffle=shuffle) 
             for device_id in range(num_workers)]
    
    for i in range(len(pipes)):
      pipes[i].build()

    self.epoch_size = pipes[0].epoch_size("Reader")
    # self.steps = -(-self.epoch_size // (len(pipes) * batch_size))
    self.steps = -(-self.epoch_size // batch_size)
    self.len = self.epoch_size if collate_fn else self.epoch_size * self.batch_size
    self.pipes = pipes
    self.records = records

    # print(self.batch_size_, batch_size, self.batch_size, self.epoch_size, self.len, self.steps)
    # print(num_workers)
    # exit(0)
    self.iter = DALIGenericIterator(pipes, pipes[0].keys, self.epoch_size, dynamic_shape=False, auto_reset=True, 
                                    fill_last_batch=False, last_batch_padded=True)
    # self.iter = DALIGenericIterator(pipes, ['id', 'index'], self.epoch_size, dynamic_shape=True, auto_reset=False)

    self.stop = False

    self.keys = self.pipes[0].keys

    self.counter = 0

    self.dataset = self
    self.sampler = None

    self.buffer = []
    self.index = 0

  def reset(self):
    for p in self.pipes:
      p.reset() 
      self.counter = 0

  def __next__(self):
    if self.counter == self.steps:
      self.reset()
      raise StopIteration

    # TODO 可以工作但是注意如果worker多个 那么可能遍历完多一些补足的样本 另外目前对应saprse的输入没有办法 只能batch_size=1 配合 dynamic_shape 读取速度非常的慢
    # https://github.com/NVIDIA/DALI/issues/1215
    # https://github.com/NVIDIA/DALI/issues/1799

    if self.index >= len(self.buffer):
      self.buffer = next(self.iter)    
      self.index = 0
    
    item = self.buffer[self.index]

    self.index += 1

    return item, item['duration']

  def __iter__(self):
    return self

  def __len__(self):
    return self.len

  def num_examples(self):
    return self.len

  def num_steps_per_epoch(self):
    return self.epoch_size
  
def get_dataset(files, td=None, shuffle=False):
  if td is None:
    return get_tfrecord_dataset(files, shuffle)
  else:
    return get_text_dataset(files, td)

def get_text_dataset(files, td):
  assert files
  datasets = [TextDataset(x, td) for x in files]
  return ConcatDataset(datasets)

def get_lmdb_dataset(files):
  assert files
  datasets = [LmdbDataset(x) for x in files]
  return ConcatDataset(datasets)

def get_tfrecord_dataset(files, shuffle=False):
  assert files
  return TFRecordDataset(files, shuffle)


if __name__=="__main__":
  pass
