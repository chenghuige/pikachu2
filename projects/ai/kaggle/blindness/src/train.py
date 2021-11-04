#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-01-13 16:32:26.966279
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np

import melt 
logging = melt.logging
import gezi
import traceback

from model import Model
from dataset import Dataset

import loss
import evaluate as ev

class TrainHook(object):
  def __init__(self):
    pass 

  def on_epoch_begin(self, epoch, model, lr):
    # if epoch == 0:
    #   lr.assign(1e-3)
    if epoch < FLAGS.num_freeze_epochs:
      model.base_model.train_able = False 
    elif epoch == FLAGS.num_freeze_epochs:
      logging.info(f'After {epoch} epochs, unfreeze base model params')
      model.base_model.train_able = True
      # lr.assign(1e-4)
    logging.info('model.base_model.train_able:', model.base_model.train_able)
    try:
      logging.info('learning rate:', lr.numpy())
    except Exception:
      pass


def main(_):
  melt.init()
  model = Model()
  logging.sinfo(model)

  fit = melt.fit

  criterion = loss.get_loss(FLAGS.loss_type)

  callbacks = [TrainHook()]

  fit(model,  
      criterion,
      Dataset,
      eval_fn=ev.evaluate,
      valid_write_fn=ev.valid_write,
      infer_write_fn=ev.infer_write,
      valid_suffix='.valid.csv',
      infer_suffix='.infer.csv',
      write_valid=True,
      callbacks=callbacks)   

if __name__ == '__main__':
  tf.compat.v1.app.run()  
