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

def loss_fn(y_true, y_pred):
  loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
  loss = mt.reduce_over(loss_func(y_true, y_pred))
  return loss

def get_loss(model=None):
  return loss_fn
