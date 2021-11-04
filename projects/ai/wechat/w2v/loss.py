#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2021-07-31 13:43:56.413495
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from icecream import ic

import tensorflow as tf
from tensorflow.keras import backend as K
import global_objectives
import melt as mt
from .config import *
  
def loss_fn(y_true, y_pred, x):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  
  loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  loss_ = loss_func(y_true, y_pred) 
  ret = mt.reduce_over(loss_)

  return ret
