#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import sys

import gezi
import melt

import tensorflow as tf

import evaluate as ev

import lele
import loss

import pyt.model as base
import torch
import text_dataset
from pyt.dataset import get_dataset
from pyt.model import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

flags = tf.app.flags
FLAGS = flags.FLAGS

logging = melt.logging

import horovod.torch as hvd
hvd.init()
torch.cuda.set_device(hvd.local_rank())

def main(_):
  FLAGS.torch_only = True
  melt.init()
  fit = melt.get_fit()

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  model = model.cuda()

  loss_fn = nn.BCEWithLogitsLoss()

  if not 'tfrecord' in FLAGS.train_input:
    td = text_dataset.Dataset()
    train_files = gezi.list_files(FLAGS.train_input)
    train_ds = get_dataset(train_files, td)
  else: 
    train_files = gezi.list_files(FLAGS.train_input)
    train_ds = get_dataset(train_files)
  
  #kwargs = {'num_workers': 4, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  #kwargs = {'num_workers': 0, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  #kwargs = {'num_workers': 4, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  
  num_workers = 1
  #kwargs = {'num_workers': num_workers, 'pin_memory': False, 'collate_fn': lele.DictPadCollate()}
  kwargs = {'num_workers': num_workers, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}

  train_sampler = train_ds
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_ds, num_replicas=hvd.size(), rank=hvd.rank())
  
  # FLAGS.batch_size is batch_size per gpu
  train_dl = DataLoader(train_ds, FLAGS.batch_size, sampler=train_sampler, **kwargs)
  
  if not 'lmdb' in FLAGS.valid_input:
    valid_files = gezi.list_files(FLAGS.valid_input)
    valid_ds = get_dataset(valid_files, td)
  else:
    valid_files = gezi.list_files(FLAGS.valid_input)
    valid_ds = get_dataset(valid_files)

  batches_per_process = -(-len(valid_ds) // hvd.size())
  start_idx = hvd.rank() * batches_per_process
  end_idx = start_idx + batches_per_process
  end_idx = min(end_idx, len(valid_ds))
  indices = range(start_idx, end_idx)
  valid_ds_per_process = torch.utils.data.Subset(valid_ds, indices)

  kwargs['num_workers'] = 2
  #valid_dl = DataLoader(valid_ds, FLAGS.eval_batch_size, sampler=valid_sampler, **kwargs)
  valid_dl = DataLoader(valid_ds_per_process, FLAGS.eval_batch_size, **kwargs)

  kwargs['num_workers'] = 1
  valid_dl2 = DataLoader(valid_ds, FLAGS.batch_size, **kwargs)


  optimizer = None
  if FLAGS.sparse_emb:
    sparse_params = list(model.wide.emb.parameters()) + list(model.deep.emb.parameters())
    ignored_params = list(map(id, sparse_params))
    sparse_params = [{'params': model.deep.emb.parameters()}, {'params': model.wide.emb.parameters(), 'lr': FLAGS.learning_rate2}]

    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    base_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer, model, num_steps_per_epoch=len(train_dl), params=base_params)
    logging.info('----------base optimizer', base_opt)
    #sparse_opt = torch.optim.SparseAdam(sparse_params)
    sparse_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer2, model, num_steps_per_epoch=len(train_dl), params=sparse_params)
    logging.info('----------sparse optimizer', sparse_opt)
    optimizer = lele.training.optimizers.MultipleOpt(base_opt, sparse_opt)
    logging.info('----------optimizer', optimizer)
  elif FLAGS.wide_deep_opt:
    wide_params = list(model.wide.parameters())
    wide_params = [{'params': model.wide.parameters(), 'lr': FLAGS.learning_rate2}]
    ignored_params = list(map(id, wide_params))
    deep_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    wide_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer, model, num_steps_per_epoch=len(train_dl), params=wide_params)
    deep_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer2, model, num_steps_per_epoch=len(train_dl), params=deep_params)
    optimizer = lele.training.optimizers.MultipleOpt(wide_opt, deep_opt)


  fit(model,  
      loss_fn,
      optimizer=optimizer,
      dataset=train_dl,
      valid_dataset=valid_dl,
      valid_dataset2=valid_dl2,
      eval_fn=ev.evaluate,
      valid_write_fn=ev.valid_write,
      #write_valid=FLAGS.write_valid)   
      write_valid=False,
     )


if __name__ == '__main__':
  tf.compat.v1.app.run()  
