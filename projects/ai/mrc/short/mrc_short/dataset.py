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
from mrc_short import util
from .config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    keys = []
    excl_keys = ['answer_start', 'answer_end', 'passage_ents', 'passage', 'title', 'query', 'answer']
    self.auto_parse(keys=keys, exclude_keys=excl_keys)
    features = self.parse_(serialized=example)

    # print('----------------------------------features', features)

    # # 如果是batch prase模式(默认) 放dataset和放到model.call差不多 放call方便debug 放这如果pytorch可以复用dataset
    # # must with usage of batch_parse, so now move to model
    # if FLAGS.batch_parse:
    #   util.adjust(features, self.subset)
      
    mt.try_append_dim(features)

    # features['answer_type_mark'] = tf.cast(features['answer_type_mark'], tf.float32)
    x = features
    y = features['passage_has_answer']

    # we need to set this since we may adjust features(change input)
    self.features = features

    return x, y
  
