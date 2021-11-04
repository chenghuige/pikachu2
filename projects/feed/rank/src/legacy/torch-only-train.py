#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pythonly-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
#   \Description  Notice pythonly train is faster then toch-train using tf dataset but it consume more resource and may fail if run 2 cocurrently
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import sys

import tensorflow as tf

import evaluate as ev
import gezi
import lele
import loss
import melt
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


def main(_):
  FLAGS.torch_only = True
  #FLAGS.valid_input = None
  melt.init()
  fit = melt.get_fit()

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  loss_fn = nn.BCEWithLogitsLoss()

  td = None

  train_files = gezi.list_files(FLAGS.train_input)
  train_ds = get_dataset(train_files, td=td)
  
  ## speed up a bit with pin_memory==True
  ## num_workers 1 is very slow especially for validation, seems 4 workers is enough, large number dangerous sometimes 12 ok sometimes hang, too much resource seems

  #kwargs = {'num_workers': 12, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  #kwargs = {'num_workers': 6, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  num_workers = 8 if gezi.get_env('num_workers') is None else int(gezi.get_env('num_workers'))
  #kwargs = {'num_workers': num_workers, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  kwargs = {'num_workers': num_workers, 'pin_memory': True, 'collate_fn': lele.NpDictPadCollate()}
  ## for 1 gpu, set > 8 might startup very slow
  #num_workers = int(8 / hvd.size())
  # num_workers = 0
  # pin_memory = False
  #kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'collate_fn': lele.DictPadCollate()}
  
  train_dl = DataLoader(train_ds, FLAGS.batch_size, shuffle=True, **kwargs)

  #kwargs['num_workers'] = max(1, num_workers)
  #logging.info('num train examples', len(train_ds), len(train_dl))

  if FLAGS.valid_input:
    valid_files = gezi.list_files(FLAGS.valid_input)
    valid_ds = get_dataset(valid_files, td)

    #kwargs['num_workers'] = 12
    valid_dl = DataLoader(valid_ds, FLAGS.eval_batch_size, **kwargs)

    #kwargs['num_workers'] = max(1, num_workers)
    valid_dl2 = DataLoader(valid_ds, FLAGS.batch_size, **kwargs)
    #logging.info('num valid examples', len(valid_ds), len(valid_dl))

  optimizer = None
  if FLAGS.sparse_emb:
    num_steps_per_epoch = len(train_dl)
    sparse_params = list(model.wide.emb.parameters()) + list(model.deep.emb.parameters())
    ignored_params = list(map(id, sparse_params))
    sparse_params = [{'params': model.deep.emb.parameters()}, {'params': model.wide.emb.parameters(), 'lr': 0.1}]

    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    base_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer, model, num_steps_per_epoch=num_steps_per_epoch, params=base_params)
    print('----------base optimizer', base_opt)
    #sparse_opt = torch.optim.SparseAdam(sparse_params)
    sparse_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer2, model, num_steps_per_epoch=num_steps_per_epoch, params=sparse_params)
    print('----------sparse optimizer', sparse_opt)
    optimizer = lele.training.optimizers.MultipleOpt(base_opt, sparse_opt)
    print('----------optimizer', optimizer)
  elif FLAGS.wide_deep_opt:
    wide_params = list(model.wide.parameters())
    wide_params = [{'params': model.wide.parameters(), 'lr': 0.1}]
    ignored_params = list(map(id, wide_params))
    deep_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    wide_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer, model, num_steps_per_epoch=num_steps_per_epoch, params=wide_params)
    print('----------wide optimizer', wide_opt)
    deep_opt = melt.eager.get_torch_optimizer(FLAGS.optimizer2, model, num_steps_per_epoch=num_steps_per_epoch, params=deep_params)
    print('----------deep optimizer', deep_opt)
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
