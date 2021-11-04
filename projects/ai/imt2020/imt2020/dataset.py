#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:11.308942
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
from tensorflow.keras import backend as K
import melt as mt
from imt2020 import util
from .config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    keys = []
    excl_keys = []
    self.auto_parse(keys=keys, exclude_keys=excl_keys)
    features = self.parse_(serialized=example)

    mt.try_append_dim(features)

    features['data'] = tf.reshape(features['data'], [-1, CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3])
    x = features['data']
    y = features['data']
    self.features = features

    return x, y
