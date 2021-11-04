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
from mrc_guwen import util
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

    features['weight'] = features['label'] * 2 + 1

    # features['answer_type_mark'] = tf.cast(features['answer_type_mark'], tf.float32)
    x = features
    y = features['label']

    # we need to set this since we may adjust features(change input)
    self.features = features

    return x, y
  
