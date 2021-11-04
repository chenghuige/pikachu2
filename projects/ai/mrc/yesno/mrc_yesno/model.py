#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:25.245765
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModel

import melt as mt
from .config import *

class Model(mt.Model):
  def __init__(self):
    super(Model, self).__init__() 
    self.input_ = None
    self.transformer = TFAutoModel.from_pretrained(FLAGS.transformer)
    self.dense = keras.layers.Dense(NUM_CLASSES) if not FLAGS.mdrop else mt.layers.MultiDropout(NUM_CLASSES)

  def call(self, input):
    self.input_ = input
    input_word_ids = input['all']
    att_mask = input_word_ids > 0
    x = self.transformer(input_word_ids, att_mask)[0]
    x = x[:, 0, :]

    if FLAGS.use_all2:
      input_word_ids2 = input['all2']
      att_mask2 = input_word_ids2 > 0
      x2 = self.transformer(input_word_ids2, att_mask2)[0]
      x2 = x2[:, 0, :]

      if FLAGS.pooling == 'concat':
        x = tf.concat([x, x2], -1)
      elif FLAGS.pooling == 'max':
        x = tf.reduce_max(tf.stack([x, x2], 1), 1)
      elif FLAGS.pooling == 'sum':
        x = tf.reduce_sum(tf.stack([x, x2], 1), 1)
      else:
        raise ValueError(FLAGS.pooling)

    logit = self.dense(x)
    if FLAGS.fp16:
      logit = tf.keras.layers.Activation('linear', dtype='float32')(logit)
    return logit

  def get_loss(self):
    from mrc_yesno import loss
    loss_fn_ = getattr(loss, FLAGS.loss)
    return self.loss_wrapper(loss_fn_)

def get_model(model_name):
  import mrc_yesno
  from .dataset import Dataset
  if model_name == 'None':
    return mt.Model()

  model = getattr(mrc_yesno.model, FLAGS.model)() 
  if FLAGS.functional_model:
    model = mt.to_functional_model(model, Dataset)

  return model
