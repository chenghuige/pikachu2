#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:33.472128
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
import melt as mt
from .config import *

def loss_fn(y_true, y_pred, x, model):
  pred = y_pred
  pred = tf.cast(pred, tf.float32)
  start = x['start_position']
  end = x['end_position']
  pred_start = pred[:, :, 0]
  pred_end = pred[:, :, 1]
  loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  start_loss = loss_func(start, pred_start)
  end_loss = loss_func(end, pred_end)
  na_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  gate_logit = model.gate_logit
  gate_logit = tf.cast(gate_logit, tf.float32)
  na_loss = na_loss_func(x['passage_has_answer'], gate_logit)
  loss = mt.reduce_over((start_loss + end_loss) * 0.5 + na_loss) 
  return loss

def get_loss(model=None):
  loss_fn_ = model.get_loss()  
  # loss_fn_ = loss_fn
  # if not FLAGS.custom_loss:
  # loss_fn_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  # else:
  #   loss_fn_ = model.get_loss()  
  return loss_fn_ 
