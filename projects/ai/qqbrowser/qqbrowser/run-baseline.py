#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:02.802049
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi as gz
logging = gz.logging
import melt as mt

import qqbrowser.eval as ev
from qqbrowser import util
from qqbrowser.config import *
from qqbrowser.loss import loss_fn_baseline as loss_fn

class Model(mt.Model):
  def __init__(self):
    super(Model, self).__init__() 
    self.input_ = None
    from baseline.tensorflow.model import MultiModal
    from baseline.tensorflow.config import parser  
    args = parser.parse_args([])
    args.num_labels = FLAGS.num_labels
    self.model = MultiModal(args)
    self.eval_keys = ['vid']
    self.str_keys = ['vid']
    self.out_keys = ['final_embedding']
    self.remove_pred = True
  
  def call(self, inputs):
    self.input_ = inputs
    preds, self.final_embedding = self.model(inputs)
    return preds

def main(_):
  timer = gz.Timer() 
  FLAGS.model_dir = '../working/baseline'
  # FLAGS.model_name = 'baseline'
  FLAGS.wandb_project = 'qqbrowser'
  FLAGS.batch_size = 512
  FLAGS.eval_batch_size = FLAGS.batch_size * 4
  FLAGS.num_gpus = -1
  FLAGS.batch_size_per_gpu = False

  FLAGS.lr = 0.005
  FLAGS.optimizer = 'adam'

  FLAGS.fp16 = False
  # FLAGS.vis = 0
  FLAGS.epochs = 8
  FLAGS.first_interval_epoch = 0.1
  FLAGS.vie = 1
  # FLAGS.mode = 'valid'
  # FLAGS.num_gpus = 1
  # FLAGS.run_eagerly = True
  FLAGS.write_valid_final = True
  FLAGS.write_valid_after_eval = True

  mt.init()

  train_dataset, (eval_dataset, val_dataset), test_dataset = util.get_datasets()
    
  fit = mt.fit  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = Model()
    fit(model,  
        loss_fn=loss_fn,
        dataset=train_dataset,
        eval_dataset=eval_dataset, 
        valid_dataset=val_dataset,
        test_dataset=test_dataset,
        eval_fn=ev.evaluate,
        valid_write_fn=ev.valid_write,
        infer_write_fn=ev.infer_write,
        ) 

if __name__ == '__main__':
  app.run(main)  
