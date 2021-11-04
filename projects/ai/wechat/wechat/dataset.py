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
from wechat import util
from wechat.config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)
    # with gezi.Timer('read history pkl'):
    if not FLAGS.static_input:
      history = gezi.read_pickle('../input/history_1.pkl')
      self.his_actions = list(list(history.values())[0].keys())
      if 'subset' == 'train':
        logging.info('his_actions', self.his_actions)

  def parse(self, example):
    keys, excl_keys = [], []
    if FLAGS.static_input:
      self.auto_parse(keys=keys, exclude_keys=excl_keys)
    else:
      # TODO better to import from history.py or config.py directly
      varlen_keys = [f'{action}s' for action in self.his_actions] + \
                    [f'{action}s_spans' for action in self.his_actions]
      self.auto_parse(keys=keys, exclude_keys=excl_keys + varlen_keys)
      self.adds(varlen_keys, tf.int64)

    # self.adds_varlens(varlen_keys)
    features = self.parse_(serialized=example)

    # print('----------------------------------features', features)

    # # 如果是batch prase模式(默认) 放dataset和放到model.call差不多 放call方便debug 放这如果pytorch可以复用dataset
    # # must with usage of batch_parse, so now move to model
    # if FLAGS.batch_parse:

    #   util.adjust(features, self.subset)

    if 'poss' in features:
      features['num_poss'] = tf.reduce_sum(tf.cast(features['poss'] > 0, tf.int32), -1)
      for action in weights_map:
        if f'{action}s' in features:
          features[f'num_{action}s'] = tf.reduce_sum(tf.cast(features[f'{action}s'] > 0, tf.int32), -1)

    mt.try_append_dim(features)

    dtype = tf.float32 if not FLAGS.fp16 else tf.float16
    # features['weight'] = tf.math.log(tf.cast(features['day'] + 1, tf.float32)) ** FLAGS.weight_power
    if FLAGS.weight_loss_byday:
      # features['weight'] = tf.cast(features['date'], tf.float32) / FLAGS.num_train_days
      features['weight'] = tf.math.log(tf.cast(features['date'] + 1, dtype)) ** FLAGS.weight_power

    if FLAGS.pos_weight:
      actions = [features[x] for x in ACTION_LIST]
      weight = tf.cast(1 + tf.reduce_sum(tf.concat(actions, -1), -1), dtype)
      features['weight'] = weight if not 'weight' in features else features['weight'] * weight

    features['finish'] = tf.minimum(features['finish_rate'], 1.) 
    features['stay'] = tf.minimum(features['stay_rate'] / 2, 1.)
    features['fresh2'] = tf.cast(features['fresh'] > 0, tf.int32)

    # features['finish_rate'] = tf.cast(features['finish_rate'] > 0.99, dtype)
    # features['stay_rate'] = tf.cast(features['stay_rate'] > 1., dtype)

    if FLAGS.his_days:
      for action in FLAGS.his_actions:
        mask = tf.cast(features[f'{action}_spans'] - 1 <= FLAGS.his_days, features[action].dtype)
        features[action] *= mask

    if 'todays_spans' not in features and 'todays' in features:
      features['todays_spans'] = tf.cast(features['todays'] > 0, tf.int32) * tf.ones_like(features['todays'], tf.int32)

    features['action'] = tf.cast(features['is_neg'] == 0, features['read_comment'].dtype)

    if FLAGS.sample_method in ['log', 'log_uniform']:
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=tf.cast(tf.reshape(features['doc'],(-1, 1)), tf.int64),  # class that should be sampled as 'positive'
        num_true=1,  # each positive skip-gram has 1 positive context class
        num_sampled=FLAGS.num_negs,  # number of negative context words to sample
        unique=True,  # all the negative samples should be unique
        range_max=vocabs['doc'].size() - 2,  # pick index of the samples from [0, vocab_size] skip 0 and 1
        seed=FLAGS.seed,  # seed for reproducibility
        name="negative_sampling"  # name of this operation
      )
      negative_sampling_candidates += 2
      negative_sampling_candidates = tf.cast(negative_sampling_candidates, features['doc'].dtype)
      features['context'] = tf.concat([features['doc'], negative_sampling_candidates], -1)

    # features['answer_type_mark'] = tf.cast(features['answer_type_mark'], tf.float32)
    x = features
    y = features['read_comment']

    # we need to set this since we may adjust features(change input)
    self.features = features

    return x, y
  
