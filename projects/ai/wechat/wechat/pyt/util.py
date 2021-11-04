#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2020-02-15 22:59:15.735957
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np

from absl import app, flags
FLAGS = flags.FLAGS

import gezi
logging = gezi.logging

import lele
import melt
from wechat.config import *
v = vocabs

def get_embedding(name, height=None, width=None, file_path=None):
  emb_flag = file_path or getattr(FLAGS, f'{name}_emb')
  if emb_flag:
    weight_file = f'../input/{emb_flag}.npy'
    weights = np.load(weight_file)[:v[name].size()]
    return nn.Embedding.from_pretrained(weights, freeze=not getattr(FLAGS, f'{name}_trainable'), padding_idx=0)
  else:
    return nn.Embedding(height or v[name].size(), width or FLAGS.emb_dim, padding_idx=0)

def get_optimizer(model, num_steps_per_epoch):
  if FLAGS.num_optimizers == 1:
    opts = [melt.eager.get_torch_optimizer(FLAGS.optimizer, model, num_steps_per_epoch=num_steps_per_epoch)]
  else:
    opt_names = FLAGS.optimizers.split(',')
    assert FLAGS.learning_rates
    logging.debug('learning_rate', FLAGS.learning_rate, 'learning_rate2', FLAGS.learning_rate2, 'opts', opt_names)
    optimizer = None
    sparse_params = []
    # if hasattr(model, 'wide'):
    #   sparse_params += list(model.wide.emb.parameters()) 
    # if hasattr(model, 'wide2'):
    #   sparse_params += list(model.wide2.emb.parameters()) 
    emb_names = ['emb', 'user_emb', 'doc_emb', 'doc_emb2']
    if hasattr(model, 'deep'):
      for name in emb_names:
        if hasattr(model.deep, name):
          sparse_params += list(getattr(model.deep, name).parameters())
    ignored_params = list(map(id, sparse_params))
    if 'emb' in FLAGS.vars_split_strategy:
      sparse_params = []
      if hasattr(model, 'deep'):
        for name in emb_names:
          if hasattr(model.deep, name):
            sparse_params += [{'params': getattr(model.deep, name).parameters(), 'lr': FLAGS.learning_rate2}]
      # if hasattr(model, 'wide'):
      #   sparse_params += [{'params': model.wide.emb.parameters(), 'lr': FLAGS.learning_rate2}] 
      # if hasattr(model, 'wide2'):
      #   sparse_params += [{'params': model.wide2.emb.parameters(), 'lr': FLAGS.learning_rate2}] 
      ## TODO dose wide with dense emb need different learning rate as FLAGS.learning_rate2 ?
      base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
      base_opt = melt.eager.get_torch_optimizer(opt_names[0], model, num_steps_per_epoch=num_steps_per_epoch, params=base_params)
      logging.debug('----------base optimizer', base_opt)
      _print_parmas(base_params)
      sparse_opt = melt.eager.get_torch_optimizer(opt_names[1], model, num_steps_per_epoch=num_steps_per_epoch, params=sparse_params)
      logging.debug('----------sparse optimizer', sparse_opt)
      _print_parmas(sparse_params)
      opts = [base_opt, sparse_opt]
    elif FLAGS.vars_split_strategy == 'wide_deep':
      # go here wide 0.01 deep 0.001
      sparse_params = []
      if hasattr(model, 'deep'):
        for name in emb_names:
          if hasattr(model.deep, name):
            sparse_params += [{'params': getattr(model.deep, name).parameters()}]
      if hasattr(model, 'wide'):
        sparse_params += [{'params': model.wide.emb.parameters(), 'lr': FLAGS.learning_rate2}] 
      if hasattr(model, 'wide2'):
        sparse_params += [{'params': model.wide2.emb.parameters(), 'lr': FLAGS.learning_rate2}] 
      base_params_deep = filter(lambda p: id(p) not in ignored_params, model.deep.parameters()) if hasattr(model, 'deep') else None
      base_params_wide = filter(lambda p: id(p) not in ignored_params, model.wide.parameters()) if hasattr(model, 'wide') else None
      base_params_wide2 = filter(lambda p: id(p) not in ignored_params, model.wide.parameters()) if hasattr(model, 'wide2') else None
      base_params = [{'params': base_params_deep}] if base_params_deep else [] + \
                    [{'params': base_params_wide, 'lr': FLAGS.learning_rate2}] if base_params_wide else [] + \
                    [{'params': base_params_wide2, 'lr': FLAGS.learning_rate2}] if base_params_wide2 else [] 
      base_opt = melt.eager.get_torch_optimizer(opt_names[0], model, num_steps_per_epoch=num_steps_per_epoch, params=base_params)
      logging.debug('----------base optimizer', base_opt)
      _print_parmas(base_params)
      sparse_opt = melt.eager.get_torch_optimizer(opt_names[1], model, num_steps_per_epoch=num_steps_per_epoch, params=sparse_params)
      logging.debug('----------sparse optimizer', sparse_opt)
      _print_parmas(sparse_params)
      opts = [base_opt, sparse_opt]
    else:
      raise ValueError(FLAGS.vars_split_strategy)
  
  if len(opts) > 1:
    optimizer = lele.training.optimizers.MultipleOpt(*opts)
  else:
    optimizer = opts[0]

  logging.debug('----------optimizer', optimizer)

  return optimizer

