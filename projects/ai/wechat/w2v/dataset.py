#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2021-07-31 08:49:52.078016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from pandas.io import feather_format
from icecream import ic
import tensorflow as tf

from gezi import tqdm
from .config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='train', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

    self.use_post_decode = FLAGS.sample_method == 'batch'

  # not correct multiple gpu...
  # 所以要从dataset提到model call处理  TODO
  def post_decode(self, x, y):
    contexts = [x['pos']]
    for _ in range(FLAGS.num_negs):  
      perm = tf.random.shuffle(tf.range(tf.shape(x['target'])[0]))
      neg = tf.gather(x['target'], perm, axis=0)
      contexts.append(neg)
    x['context'] = tf.expand_dims(tf.stack(contexts, 1), -1)
    # x (bs, 1 + num_negs, 1)
    return x, y

  def parse(self, example):
    keys, excl_keys = [], []
    self.auto_parse(keys=keys, exclude_keys=excl_keys)
    fe = self.parse_(serialized=example)

    if FLAGS.sample_method == 'log_uniform':
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=tf.cast(tf.reshape(fe['pos'],(-1, 1)), tf.int64) - FLAGS.start_id,  # class that should be sampled as 'positive'
        num_true=1,  # each positive skip-gram has 1 positive context class
        num_sampled=FLAGS.num_negs,  # number of negative context words to sample
        unique=True,  # all the negative samples should be unique
        range_max=vocabs[FLAGS.attr].size() - FLAGS.start_id,  # pick index of the samples from [0, vocab_size]
        seed=FLAGS.seed,  # seed for reproducibility
        name="negative_sampling"  # name of this operation
      )
      negative_sampling_candidates += FLAGS.start_id
      negative_sampling_candidates = tf.cast(negative_sampling_candidates, fe['pos'].dtype)
      fe['context'] = tf.concat([tf.reshape(fe['pos'], (-1, 1)), tf.reshape(negative_sampling_candidates, (-1, FLAGS.num_negs))], -1)
 
    mt.try_append_dim(fe)
    x = fe
    # TODO should be fe['target'][0] ? the same ? [bs, 1] or [bs]
    y = tf.zeros_like(fe['target'], dtype=tf.float32)
    fe['y'] = y

    self.features = fe

    return x, y
  
