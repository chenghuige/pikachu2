#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-26 23:00:24.215922
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

from tensorflow.keras import backend as K

import melt as mt
import gezi
logging = gezi.logging
import numpy as np
import time
import datetime

from projects.feed.rank.src.config import *
from projects.feed.rank.src.util import * 


def adjust(features, subset, embedding_keys=None):
  features['click'] = tf.cast(K.not_equal(features['duration'], 0), tf.int64)

  if FLAGS.use_position_emb:
    if 'position' in features and subset != 'train':
      features['position'] = tf.zeros_like(features['duration'])

  if 'read_completion_rate' not in features:
    features['read_completion_rate'] = tf.zeros_like(features['duration'], dtype=tf.float32)
    
  ## Already done when gen tfrecord
  # if FLAGS.is_video and FLAGS.cut_duration_by_video_time:
  #   assert 'video_time' in features
  #   features['duration'] = tf.math.minimum(features['duration'], features['video_time'])
                  
  # TODO duration tf.cond only ok if not batch parsing.. here just comment out since we have done this on preparing tfrecord, Try to add support for batch parsing 
  # features['duration'] = tf.cond(pred=features['duration'] > 60 * 60 * 12, true_fn=lambda: tf.constant(60, dtype=tf.int64), false_fn=lambda: features['duration'])
  
  #------might change duraiton to 0  only for train
  if FLAGS.min_click_duration > 1 and subset == 'train':
    if not FLAGS.new_duration:
      mask = tf.cast(features['duration'] >= FLAGS.min_click_duration, tf.int64)
      features['click'] = features['click'] * mask
      features['duration'] = features['duration'] * mask 
    else:
      mask = tf.cast(features['duration'] < FLAGS.min_click_duration, tf.int64)
      fdur_bad = features['duration'] * mask
      fdur_ok = features['duration'] * (1 - mask)
      features['duration'] = fdur_ok + tf.cast(tf.cast(fdur_bad, tf.float32) * 0.1, tf.int64)  
  
  #------notice features always has weight but you can choose not to use it by setting FLAGS.use_weight=0 by default will be True
  features['weight'] = 1.
  logging.debug('duration_weight', FLAGS.duration_weight, 'min_click_duration', FLAGS.min_click_duration)
  #-----------set weight according to duration, also change valid weight as want valid loss similar strategy as train
  if FLAGS.duration_weight:
    weight = tf.cast(features['duration'], tf.float32)
    weight = tf.math.minimum(weight, FLAGS.max_duration)
    if FLAGS.duration_weight_nolog:
      # ratio = weight / FLAGS.max_duration
      # weight = 5. * (ratio / tf.math.sqrt(1. + ratio ** 2.))
      features['weight'] = weight * FLAGS.duration_weight_multiplier + 1.
    else:
      features['weight'] = (tf.math.log(weight + 1.) ** FLAGS.duration_weight_power) * FLAGS.duration_weight_multiplier
      features['weight'] += tf.cast(1 - features['click'], tf.float32)
    if FLAGS.duration_ratio < 1.:
      ratio = FLAGS.duration_ratio
      features['weight'] = features['weight'] * ratio + (1 - ratio)

  # Notice weight only affect valid loss not other evaluations(they all not use weight)
  logging.debug('interests_weight', FLAGS.interests_weight, 'type', FLAGS.interests_weight_type)
  if FLAGS.interests_weight:
    if FLAGS.interests_weight_type == 'ori':
      w = tf.cast(features['num_interests'] + 1, tf.float32) / 100.
      w = tf.math.minimum(w, 10.0)
    elif FLAGS.interests_weight_type == 'log':
      w = tf.math.log(tf.cast(features['num_interests'], tf.float32) + 1.1)
    elif  FLAGS.interests_weight_type == 'clip':
      w = tf.cast(features['num_interests'] >= FLAGS.min_interests, tf.float32)
    elif FLAGS.interests_weight_type == 'drop':
      w = tf.cast(features['num_interests'] < FLAGS.min_interests, tf.float32) * FLAGS.min_interests_weight + tf.cast(features['num_interests'] >= FLAGS.min_interests, tf.float32)
    else:
      raise ValueError("Not support" + FLAGS.interests_weight_type)
    
    features['weight'] *= w

  logging.debug('min_filter_duration', FLAGS.min_filter_duration)
  # filter like < 5s duration instances for train only, notice set weight not affect valid evaluation only affect valid loss
  if FLAGS.min_filter_duration > 1:
    w = tf.cast(features['duration'] >= FLAGS.min_filter_duration, tf.float32)
    # free if is special click, wich means wo do not know exacatly dur time, but they are click
    w += tf.cast(features['duration'] < 0, tf.float32)
    # for tuwen since video_time is 0 will get finish_ration nearly as 0 which will <  min_finish_ratio
    if FLAGS.min_finish_ratio > 0. and 'video_time' in features: 
      finish_ratio = get_finish_ratio(features)
      # free if finish ratio ok
      w += tf.cast(finish_ratio >= FLAGS.min_finish_ratio, tf.float32)
    w = tf.cast(w > 0, tf.float32)
    neg_w = tf.cast(K.equal(features['duration'], 0), tf.float32)
    pos_w = 1 - neg_w 
    w = w * pos_w + neg_w
    features['weight'] *= w

  if FLAGS.filter_neg_durations:
    w = tf.cast(features['duration'] >= 0, tf.float32)
    features['weight'] *= w

  # for cold start user weight * 0.1 for train for eval will consider exclude cold and also cold only 
  # NOTICE only neg examples of cold start * 0.1 for pos example just 1.
  features['cold'] = tf.cast(is_cb_user(features['rea']), tf.int64)
  if FLAGS.torch:
    features['product_id'] = tf.cast(get_product_id(features['product']), tf.int64)
  if FLAGS.change_cb_user_weight:
    rea_w = tf.cast(features['cold'], tf.float32)
    neg_w = tf.cast(K.equal(features['duration'], 0), tf.float32)
    rea_w *= neg_w
    nonrea_w = 1 - rea_w
    rea_w *= FLAGS.cb_user_weight
    w = rea_w + nonrea_w
    features['weight'] *= w

  logging.debug('weight', features['weight'], 'subset', subset)

  # from [batch_size,] to [batch_size,1] mainly for compat with keras and not keras as keras will auto add 1
  mt.try_append_dim(features)
  # mt.try_append_dim(features, ['click', 'duration'])

  features['click'] = tf.cast(features['click'], tf.float32) 

  # del features['click']

  if not 'tw_history' in features:
    features['tw_history'] = features['history']
    features['vd_history'] = features['history']

  if 'doc_title' in features:
    features['title'] = features['doc_title']

  if embedding_keys:
    features['index'] = {}
    features['value'] = {}
    features['field'] = {}
    for key in embedding_keys:
      features['index'][key] = features[f'index_{key}']
      del features[f'index_{key}']
      if f'value_{key}' in features:
        features['value'][key] = features[f'value_{key}']
        del features[f'value_{key}']

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)
    self.all_varlen_keys = []
    self.varlen_keys = []

    # valen.txt generated by tools/stats-tfrecord.py
    for line in open('./conf/varlen.txt'):
      key, _, _ = line.strip().split()
      if key.startswith('#'):
        key = key[1:]
      else:
        self.varlen_keys.append(key)
      self.all_varlen_keys.append(key)

  def parse(self, example):
    keys = []
    if FLAGS.all_varlen_keys:
      self.adds_varlens()
    elif FLAGS.record_padded:
      self.auto_parse(keys=keys)
    else:
      self.auto_parse(keys=keys, exclude_keys=self.all_varlen_keys)
      if not FLAGS.exclude_varlen_keys:
        self.adds_varlens(self.all_varlen_keys)
    # value might be treated as index change to float
    self.adds(['value'], tf.float32)

    self.embedding_keys = None

    features = self.parse_(serialized=example)    
    adjust(features, self.subset, self.embedding_keys)

    x = features
    y = features['click']

    logging.debug('x', x, 'y', y, 'subset', self.subset)

    return x, y
