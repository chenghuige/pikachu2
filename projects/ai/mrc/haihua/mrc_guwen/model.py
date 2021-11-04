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
from transformers import TFAutoModel, BertConfig

import melt as mt
from .config import *

class Model(mt.Model):
  def __init__(self):
    super(Model, self).__init__() 
    self.input_ = None
    if not os.path.exists(FLAGS.transformer):
      self.transformer = TFAutoModel.from_pretrained(FLAGS.transformer, from_pt=FLAGS.from_pt)
    else:
      config = BertConfig.from_json_file(f'{FLAGS.transformer}/bert_config.json')
      self.transformer = TFAutoModel.from_pretrained(FLAGS.transformer, from_pt=FLAGS.from_pt, config=config)

    Dense = keras.layers.Dense if not FLAGS.mdrop else mt.layers.MultiDropout
    self.dense = Dense(1)

  def call(self, input):
    self.input_ = input
    input_mask = tf.cast(input['input_ids'] > 0, input['input_ids'].dtype)
    x = self.transformer(input['input_ids'], input_mask, input['segment_ids'])[0][:,0]
    logit = self.dense(x)
    return logit

  def get_loss(self):
    from mrc_guwen import loss
    loss_fn_ = getattr(loss, FLAGS.loss)
    return self.loss_wrapper(loss_fn_)

def get_model(model_name):
  import mrc_guwen
  from .dataset import Dataset
  if model_name == 'None':
    return mt.Model()

  model = getattr(mrc_guwen.model, FLAGS.model)() 
  if FLAGS.functional_model:
    model = mt.to_functional_model(model, Dataset)

  return model
