#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2018-10-17 06:52:08.997327
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import flags
FLAGS = flags.FLAGS

import tensorflow as tf
import torch
from torch import nn
#from torch.utils.data import Dataset, ConcatDataset

import copy
import traceback
import numpy as np

import gezi 
logging = gezi.logging

def adjust_lrs(x, ratio=None, name='learning_rate_weights'):
  import tensorflow as tf
  if ratio is None:
    ratios = tf.compat.v1.get_collection(name)[-1].numpy()
    # TODO will this hurt performance ? change to use learning rate weights without tf dependence?
    ratios = torch.as_tensor(ratios).cuda()
    x = x * ratios + x.detach() * (1 - ratios)
  else:
    x = x * ratio + x.detach() * (1 - ratio)
  return x 


def load(model, path):
  try:
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']   
    
    model_ = model.module if hasattr(model, 'module') else model
    new_state = {}
    for key, val in state.items():
      if key in model_.state_dict():
        new_state[key] = val

    logging.info('Updated %d keys from checkpoint %s, eopoch:%d, step:%d' % (len(new_state), path, checkpoint['epoch'], checkpoint['step']))
    new_params = model_.state_dict()
    new_params.update(new_state)
    model_.load_state_dict(new_params)
    
    model.eval()

    updated_params = []
    for name, param in model_.named_parameters():
      if name in new_state:
        updated_params.append(param)

    return checkpoint, updated_params 
  except Exception:
    logging.info(traceback.print_exc())
    return None, []

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

try:
  import torch 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
  pass
import numpy as np 

def torch_(x, cuda=True):
  global device
  if FLAGS.torch_only:
    return x
  for dim in x.shape:
    if dim == 0:
      return x

  # if tf.__version__ < '2':
  x = x.numpy()

  device = gezi.get('device') or device

  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.as_tensor(x)
    if cuda:
      x = x.to(device)

  return x

def to_torch(x, y=None, cuda=True, torch_only=False):
  global device
  if torch_only or FLAGS.torch_only:
    if cuda:
      device = gezi.get('device') or device
      for key in x:
        if type(x[key]) != np.ndarray and not isinstance(x[key], (list, tuple)):
          x[key] = x[key].to(device)
      return x, y.to(device)
    else:
      return x, y

  if y is not None:
    y = torch_(y, cuda)

  if not isinstance(x, dict):
    x = torch_(x, cuda)
  else:
    for key in x:
      x[key] = to_torch(x[key], cuda=cuda)
      
  if y is None:
    return x
  else:
    return x, y

#---------------padding input data

#https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/12

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)

class PadCollate2:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([torch.Tensor(x[0]).shape[self.dim] for x in batch])
        #print('----------', max_len)
        # pad according to max_len
        batch = [(pad_tensor(torch.Tensor(x[0]), pad=max_len, dim=self.dim), x[1]) for x in batch]
        # stack all
        xs = torch.stack([torch.Tensor(x[0]) for x in batch], dim=0)
        ys = torch.Tensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
      
class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([x[0].size(self.dim) for x in batch])
        #print('----------', max_len)
        # pad according to max_len
        batch = [(pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]) for x in batch]
        # stack all
        xs = torch.stack([x[0] for x in batch], dim=0)
        ys = torch.Tensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

class NpDictPadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
      ys = [None] * len(batch)
      input = {}
      ys[0] = batch[0][1]
      max_len = {}
      
      for key, val in batch[0][0].items():
        if isinstance(val, np.ndarray):
          val = torch.as_tensor(val)
          max_len[key] = len(val)
        else:
          if isinstance(val, list):
            if type(val[0]) == int:
              val = torch.as_tensor(np.asarray(val))
            else:
              val = torch.as_tensor(np.asarray(val)).float()
            max_len[key] = len(val)
        input[key] = [val] * len(batch)
       
      for i in range(1, len(batch)):
        ys[i] = batch[i][1]
        for key, val in batch[i][0].items():
          if isinstance(val, np.ndarray):
            val = torch.as_tensor(val)
            if len(val) > max_len[key]:
              max_len[key] = len(val)
          else:
            if isinstance(val, list):
              if type(val[0]) == int:
                val = torch.as_tensor(np.asarray(val))
              else:
                val = torch.as_tensor(np.asarray(val)).float()
              if len(val) > max_len[key]:
                max_len[key] = len(val)
          input[key][i] = val
          
      for key, val_list in input.items():
        if key in max_len:
          for i in range(len(val_list)):
            val_list[i] = pad_tensor(val_list[i], pad=max_len[key], dim=self.dim)
            #print(i, val_list[i].shape, max_len[key])
    
          input[key] = torch.stack(val_list, dim=0)
        else:
          #... TODO why np.arry.dtype not dp.str_ but <U3 <U4 ?
          input[key] = np.asarray(input[key])
          if type(input[key][0]) != np.str_:
            input[key] = torch.as_tensor(input[key])
            
      ys = torch.as_tensor(np.asarray(ys))
      return input, ys
        
    def __call__(self, batch):
        return self.pad_collate(batch)
      
class DictPadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
      ys = [None] * len(batch)
      input = {}
      ys[0] = batch[0][1]
      max_len = {}
      
      for key, val in batch[0][0].items():
        #if not isinstance(val, str):
        if isinstance(val, torch.Tensor):
          if not len(val.size()):
            val = val.expand(1)
          max_len[key] = val.size(self.dim)
        input[key] = [val] * len(batch)
       
      for i in range(1, len(batch)):
        ys[i] = batch[i][1]
        for key, val in batch[i][0].items():
          #if not isinstance(val, str):
          if isinstance(val, torch.Tensor):
            if not len(val.size()):
              val = val.expand(1)
            if len(val) > max_len[key]:
              max_len[key] = val.size(self.dim)
          input[key][i] = val
          
      for key, val_list in input.items():
        if key in max_len:
          for i in range(len(val_list)):
            val_list[i] = pad_tensor(val_list[i], pad=max_len[key], dim=self.dim)  
          input[key] = torch.stack(val_list, dim=0)
        else:
          input[key] = np.array(input[key])

      #list of tensor ->
      ys = torch.stack(ys, dim=0)
      return input, ys
        
    def __call__(self, batch):
      return self.pad_collate(batch)

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def keras_init(model, emb=True, linear=False):
  for m in model.modules():
    if emb:
      if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
        if m.weight.requires_grad:
          logging.debug(m, 'keras init emb')
          nn.init.uniform_(m.weight, -0.05, 0.05)
    if linear:
      if isinstance(m, nn.Linear):
        logging.debug(m, 'keras init linear')
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class PytObj(object):
  def __init__(self, x):
    self.x = x

  def numpy(self):
    return self.x

class PytMean(object):
  def __init__(self):
    self._val = 0. 
    self.count = 0

    self.is_call = True

  def clear(self):
    self._val = 0
    self.count = 0

  def __call__(self, val=None):
    if val is None:
      return self.result()
    if not self.is_call:
      self.clear()
      self.is_call = True
    self._val += val.item()
    self.count += 1

  def result(self):
    if self.is_call:
      self.is_call = False
    if not self.count:
      val = 0
    else:
      val = self._val / self.count
    # TODO just for compact with tf ..
    return PytObj(val)

  def numpy(self):
    return self.result().numpy()
  