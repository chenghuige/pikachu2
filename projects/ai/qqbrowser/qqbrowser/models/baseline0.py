#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   baseline0.py
#        \author   chenghuige  
#          \date   2021-08-26 04:53:20.778220
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import melt as mt
from ..config import *

class Model(mt.Model):
  def __init__(self):
    super(Model, self).__init__() 
    self.input_ = None
    from baseline.tensorflow.model import MultiModal
    from baseline.tensorflow.config import parser  
    args = parser.parse_args([])
    args.num_labels = FLAGS.num_labels
    self.model = MultiModal(args)
  
  def call(self, inputs):
    self.input_ = inputs
    if not 'input_ids' in inputs:
      inputs['input_ids'] = inputs['title_ids']
      inputs['mask'] = inputs['title_mask']
    preds, self.final_embedding = self.model(inputs)
    return preds
  
  def get_loss(self):
    loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def loss_fn(y_true, y_pred):
      loss = loss_obj(y_true, y_pred)
      loss *= y_true.shape[-1]
      return mt.reduce_over(loss)
    
    return self.loss_wrapper(loss_fn)
  