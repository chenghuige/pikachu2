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
from projects.ai.ad.src.config import *

# {'id': array([10]), 'age': array([9]), 'ad_ids': array([ 121278,   63976,  177187,   74359, 2866446,   66208,   66208,
#        1410336,   40546,  375851]), 'creative_ids': array([ 134938,   69204,  197464,   81006, 3331781,   71689,   71689,
#        1626693,   42577,  421621]), 'industries': array([ 60,  60,  60, 158,   6,   6,   6, 318, 231, 231]), 'advertiser_ids': array([62956, 24952,  2367,   953, 52604, 14681, 14681, 22885, 18230,
#        18230]), 'product_categories': array([ 3,  3,  3, 18,  2, 18, 18,  2,  2,  2]), 'times': array([ 4,  7, 22, 28, 70, 73, 74, 79, 84, 91]), 'gender': array([2]), 'click_times': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}

class Dataset(melt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    features_dict = {}

    LEN = FLAGS.max_len

    def _adds(names, dtype=None, length=None):
      self.adds(features_dict, names, dtype, length)

    # _adds(['id'], tf.int64, 0)
    _adds(['age', 'gender'], tf.int64, 1)
    _adds(['creative_ids', 'ad_ids', 'industries', 'advertiser_ids', 'product_ids', 'product_categories', 'times', 'click_times'], tf.int64, LEN)

    features = self.parse_(serialized=example, features=features_dict)

    features['gender'] -= 1
    # if 'Cls' in FLAGS.model:
    features['age'] -= 1
    # else:
    #   features['age'] = tf.cast(features['age'] - 1, tf.float32) / 10.

    features['id'] = tf.squeeze(tf.zeros_like(features['age'], tf.string))
    x = features
    # if not FLAGS.lm_target:
    y = tf.cast(features['age'], tf.int64)
    # else: 
    #   y = features[FLAGS.lm_target]
    #   if FLAGS.lm_target in ['ad_ids', 'creative_ids']:
    #     mask = tf.cast(features[FLAGS.lm_target] < 1000000, tf.int64)
    #     mask2 = 1 - mask
    #     y = y * mask + mask2
    #   y = y[:,:LEN]

    return x, y
    