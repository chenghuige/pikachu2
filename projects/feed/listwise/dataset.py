#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2020-04-12 20:33:51.902319
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
import melt

class Dataset(melt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    features_dict = {}

    def _adds(names, dtype=None, length=None):
      self.adds(features_dict, names, dtype, length)

    _adds(['impression'], tf.string, 0)
    _adds(['durs', 'positions'], tf.int64)
    _adds(['click_feats', 'dur_feats'], tf.float32)

    features = self.parse_(serialized=example, features=features_dict)

    features['id'] = features['impression']

    x = features
    y = tf.cast(K.not_equal(features['durs'], 0), tf.float32)

    return x, y
    