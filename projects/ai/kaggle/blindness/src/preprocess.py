#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige  
#          \date   2019-09-03 07:08:29.200609
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys 
import os

import numpy as np

from config import *

def to_regression(y):
  bs = len(y)
  y = np.argmax(y, -1).reshape([bs, 1])
  return y 

def to_regression2(y):
  bs = len(y)
  y = np.argmax(y, -1).reshape([bs, 1])
  y = y / (NUM_CLASSES - 1)
  return y 

def to_ordinal(y):
  # [0,0,1,0,0] ->[1,1,1,0,0] for multi label loss
  y_ = np.empty(y.shape, dtype=y.dtype)
  y_[:, NUM_CLASSES - 1] = y[:, NUM_CLASSES - 1]

  for i in range(NUM_CLASSES - 2, -1, -1):
      y_[:, i] = np.logical_or(y[:, i], y_[:, i+1])
  y = y_  
  return y

def to_ordinal2(y):
  y = to_ordinal(y)
  y = y[:, 1:]
  return y

def trans_y(y, loss_type):
  if 'regression' in loss_type:
    if 'sigmoid2' in loss_type:
      return to_regression2(y)
    else:
      return to_regression(y)
  elif 'ordinal' in loss_type:
    if '2' in loss_type:
      return to_ordinal2(y)
    else:
      return to_ordinal(y)
  return y