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
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
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
