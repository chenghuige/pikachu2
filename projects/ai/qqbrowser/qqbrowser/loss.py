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
from math import gamma

import sys 
import os

import tensorflow as tf
from tensorflow.keras import backend as K
import melt as mt
from .config import *

def loss_fn_baseline(y_true, y_pred):
  loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
  loss = loss_obj(y_true, y_pred)
  return mt.reduce_over(loss)

def get_loss(model=None):
  loss_fn_ = model.get_loss()  
  return loss_fn_ 
