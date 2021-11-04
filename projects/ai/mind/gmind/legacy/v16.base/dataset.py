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
import melt as mt
from projects.ai.mind.src import util
from projects.ai.mind.src.config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

    # if not FLAGS.batch_parse:
    #   # 只能非batch_parse模式 而且速度很慢 eager验证正确 但是batch_parse当前对varlen key某个有bug TODO feed rank数据没问题
    #   # 因为速度很慢 所以 filter基本很不实用了 本身batch_prase=0速度就会变慢
    #   def filter(x, y):
    #     ok = tf.cast(tf.cast(K.greater_equal(tf.random.uniform(shape=[]), FLAGS.neg_filter_ratio), tf.int64) + tf.squeeze(x['click']), tf.bool)
    #     return ok
    #   self.filter_fn = filter if subset == 'train' else None

  def parse(self, example):
    keys = []
    if FLAGS.record_padded:
      self.auto_parse(keys=keys)
    else:
      varlen_keys = [
            'history', 
            'impressions',
            
            'history_cats', 'history_sub_cats', 
            'history_title_entities', 'history_abstract_entities',
            'history_title_entity_types', 'history_abstract_entity_types',
            'title_entities', 'abstract_entities',
            'title_entity_types', 'abstract_entity_types',
           ]
      self.auto_parse(keys=keys, exclude_keys=varlen_keys)
      if not FLAGS.exclude_varlen_keys:
        self.adds(varlen_keys, tf.int64)

    features = self.parse_(serialized=example)

    # 如果是batch prase模式(默认) 放dataset和放到model.call差不多 放call方便debug 放这如果pytorch可以复用dataset
    # must with usage of batch_parse, so now move to model
    if FLAGS.batch_parse:
      util.adjust(features, self.subset)
      
    # 注意parse出来单独的 (,) 不带（,1)
    ## 之前keras模式需要append dim, 现在看起来tf2 keras会自动append  但是tf1.15 keras模式目前无法跑 pikachu/examples/tf/imdb/keras-train-tfrecord.py
    ## ValueError: For performance reasons Keras `fit`, `evaluate` and`predict` accept tf.data `Datasets` as input
    ## but not iterators that have been manually generated from Datasets by users. 
    ## Please directly pass in the original `Dataset` object instead of passing in `iter(dataset)`.
    mt.try_append_dim(features)

    # mt.try_append_dim(features, ['click'])
    features['click'] = tf.cast(features['click'], tf.float32)
    x = features
    y = features['click']

    if FLAGS.lm_target:
      y = features[FLAGS.lm_target]

    # we need to set this since we may adjust features(change input)
    self.features = features

    return x, y
    
