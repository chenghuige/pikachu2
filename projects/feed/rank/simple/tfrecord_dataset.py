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
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt 
logging = melt.logging
import numpy as np

from config import *

class Dataset(melt.Dataset):
  def __init__(self, subset='valid'):
    super(Dataset, self).__init__(subset)
  
  def parse(self, example):
    features_dict = {
      'id':  tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64),
      'index': tf.io.VarLenFeature(tf.int64),
      'field': tf.io.VarLenFeature(tf.int64),
      'value': tf.io.VarLenFeature(tf.float32),
      'doc_emb': tf.io.FixedLenFeature([FLAGS.doc_emb_dim], tf.float32)
      }

    features = tf.io.parse_single_example(serialized=example, features=features_dict)

    logging.info('features', features)

    y = features['label']
    y = tf.cast(y, tf.float32)
    del features['label']

    x = features

    return x, y
