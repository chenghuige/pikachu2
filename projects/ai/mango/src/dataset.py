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
from projects.ai.mango.src.config import *
from projects.ai.mango.src.util import *

class Dataset(melt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    features_dict = {}

    def _adds(names, dtype=None, length=None):
      self.adds(features_dict, names, dtype, length)

    _adds(['did_'], tf.string, 0)
    _adds(['vid_'], tf.int64, 0)
    _adds(['index'], tf.int64, 0)
    _adds(['label', 'day'], tf.int64, 1)
    _adds(['region'], tf.int64, 0)
    _adds(['mod', 'mf', 'sver', 'aver'], tf.int64, 0)
    _adds(['hour', 'weekday'], tf.int64, 0)
    _adds(['fresh', 'title_length'], tf.int64, 0)
    _adds(['ctr', 'vv', 'duration', 'cid_rate'], tf.float32, 0)
    _adds(['cid', 'class_id', 'second_class', 'is_intact'], tf.int64, 0)
    _adds(['did', 'vid', 'prev', 'has_prev'], tf.int64, 0)
    _adds(['watch_vids', 'watch_times'], tf.int64, 50)
    _adds(['show_vids'], tf.int64, 50)
    _adds(['stars'], tf.int64)
    _adds(['timestamp', 'hits', 'num_shows'], tf.int64, 0)
    _adds(['title', 'story'], tf.int64)
    _adds(['titles'], tf.int64)
    _adds(['stars_list', 'all_stars_list'], tf.int64)
    _adds(['first_stars_list'], tf.int64, 50)
    _adds(['durations'], tf.float32, 50)
    _adds(['freshes'], tf.int64, 50)
    # TODO should be last_title.. spell err

    # if 'v9' in FLAGS.train_input:
    #   _adds(['last_tile', 'last_stars'], tf.int64)
    # else:
    #   _adds(['last_title', 'last_stars'], tf.int64)
    
    # _adds(['image_emb'], tf.float32)
    _adds(['cids', 'class_ids', 'second_classes', 'is_intacts'], tf.int64, 50)

    _adds(['ctr_', 'vv_', 'title_length_', 'duration_'], tf.int64, 0)

    _adds(['prev_duration', 'prev_ctr'], tf.float32, 0)
    _adds(['prev_is_intact', 'prev_title_length', 'prev_vv', 'prev_duration_', 
           'prev_title_length_', 'prev_ctr_', 'prev_vv_'], tf.int64, 0)

    if FLAGS.use_contexts:
      for i in range(len(context_cols) - 1):
        for j in range(i + 1, len(context_cols)):
          _adds([f'{context_cols[i]}_{context_cols[j]}'], tf.int64, 0)

    if FLAGS.use_items:
      for i in range(len(item_cols) - 1):
        for j in range(i + 1, len(item_cols)):
           _adds([f'{item_cols[i]}_{item_cols[j]}'], tf.int64, 0)

    if FLAGS.use_crosses:
      for context_col in context_cols:
        for item_col in item_cols:
           _adds([f'cross_{context_col}_{item_col}'], tf.int64, 0)

    features = self.parse_(serialized=example, features=features_dict)

    features['id'] = features['index']

    # if 'v9' in FLAGS.train_input:
    #   features['last_title'] = features['last_tile']
    #   del features['last_tile']

    if FLAGS.use_weight:
      # features['weight'] = tf.math.log(tf.cast(features['day'] + 1, tf.float32)) ** FLAGS.weight_power
      # mask = tf.cast(tf.equal(features['index'], -1), tf.float32)
      # features['weight'] =  mask * 0.1 + (1 - mask)
      if K.learning_phase() == 1:
        mask = tf.cast(tf.equal(features['label'], 1), tf.float32)
        features['weight'] = tf.math.log(tf.cast(features['num_shows'], tf.float32)) * mask + (1 - mask)
      else:
        features['weight'] = 1.

    if FLAGS.use_contexts:
      feats = []
      for i in range(len(context_cols) - 1):
        for j in range(i + 1, len(context_cols)):
          feats.append(features[f'{context_cols[i]}_{context_cols[j]}'])    
      features['context'] = tf.stack(feats, axis=1)

    if FLAGS.use_items:
      feats = []
      for i in range(len(item_cols) - 1):
        for j in range(i + 1, len(item_cols)):
          feats.append(features[f'{item_cols[i]}_{item_cols[j]}'])    
      features['item'] = tf.stack(feats, axis=1)

    if FLAGS.use_crosses:
      feats = []
      for context_col in context_cols:
        for item_col in item_cols:
          if context_col in ignored_cols and item_col in ignored_cols:
            continue
          # try:
          feats.append(features[f'cross_{context_col}_{item_col}'])    
          # except Exception:
            # pass
      features['cross'] = tf.stack(feats, axis=1)

    if FLAGS.use_unk:
      max_vid = gezi.get('vocab_sizes')['vid'][1]
      features['vid'] = get_vid(features['vid'], max_vid)
      features['watch_vids'] = get_vid(features['watch_vids'], max_vid)

    x = features
    y = features['label']

    if FLAGS.lm_target:
      y = features[FLAGS.lm_target]

    return x, y
    
