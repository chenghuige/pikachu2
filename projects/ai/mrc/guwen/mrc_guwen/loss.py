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
  loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  loss = loss_func(y_true, pred)
  if FLAGS.use_weight:
    weights = tf.cast(x['weight'], tf.float32)
    loss *= weights
    loss /= tf.reduce_sum(weights)
  loss = mt.reduce_over(loss)
  return loss

def get_loss(model=None):
  loss_fn_ = model.get_loss()  
  # loss_fn_ = loss_fn
  # if not FLAGS.custom_loss:
  # loss_fn_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  # else:
  #   loss_fn_ = model.get_loss()  
  return loss_fn_ 
