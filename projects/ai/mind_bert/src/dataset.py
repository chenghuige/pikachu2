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
import melt
from config import *
from utils import *

class Dataset(melt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

    # @tf.function
    def undersampling_filter(x, y):
      prob = tf.cond(pred=tf.equal(x['src'], 'test'), true_fn=lambda: 1., false_fn=lambda: FLAGS.sampling_rate)
      acceptance = tf.less_equal(tf.random.uniform([], dtype=tf.float32), prob)
      return acceptance
    
    self.filter_fn = undersampling_filter if FLAGS.sampling_rate < 1 else None

  def parse(self, example):
    MAX_LEN = FLAGS.max_len
    features_dict = {}

    def _adds(names, dtype=None, length=None):
      dtype_ = dtype
      for name in names:
        if name in self.example:
          dtype = dtype_ or self.example[name].dtype 
          if length is None:
            features_dict[name] = tf.io.VarLenFeature(dtype)
          elif length > 0:
            features_dict[name] = tf.io.FixedLenFeature([length], dtype)
          else:
            features_dict[name] = tf.io.FixedLenFeature([], dtype)

    _adds(['toxic', *toxic_types])
    _adds(['input_word_ids', 'input_word_ids2'], tf.int64, MAX_LEN)
    _adds(['id', 'lang', 'src'], tf.string, 0)
    _adds(['trans'], tf.int64, 1)
    _adds(['input_mask', 'all_segment_id'], tf.int64, MAX_LEN)
    
    features = self.parse_(serialized=example, features=features_dict)

    features['lang_'] = tf.expand_dims(get_lang_id(features['lang']), -1)
    features['src_'] = tf.expand_dims(get_src_id(features['src']), -1)

    def _casts(names, dtype=tf.int32):
      for name in names:
        if name in features:
          features[name] = tf.cast(features[name], dtype)

    # if not FLAGS.torch:
    #   _casts(['input_word_ids', 'input_word_ids2', 'input_mask', 'all_segment_id', 'trans'])

    x = features

    if FLAGS.task == 'toxic':
      y = features['toxic']
      keys = ['toxic', *toxic_types]
      for key in keys:
        if key not in features:
          features[key] = tf.zeros_like(features['toxic'])
          
      _casts(toxic_types, tf.float32)
          
      melt.append_dim(features, keys)

      if FLAGS.multi_head:
        y = tf.concat([features[key] for key in keys], 1)

    elif FLAGS.task == 'lang':
      y = tf.one_hot(features['lang_'], len(langs))

    return x, y
    