#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   utils.py
#        \author   chenghuige  
#          \date   2020-04-21 15:45:28.877753
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from config import *
  
# langs = ['en', 'es', 'it', 'tr', 'fr', 'pt', 'ru']

def get_lang_id(x):
  return tf.cast(tf.math.equal(x, 'en'), tf.int32) \
         + tf.cast(tf.math.equal(x, 'es'), tf.int32) * 2 \
         + tf.cast(tf.math.equal(x, 'it'), tf.int32) * 3 \
         + tf.cast(tf.math.equal(x, 'tr'), tf.int32) * 4 \
         + tf.cast(tf.math.equal(x, 'fr'), tf.int32) * 5 \
         + tf.cast(tf.math.equal(x, 'pt'), tf.int32) * 6 \
         + tf.cast(tf.math.equal(x, 'ru'), tf.int32) * 7 \
         - 1

# srcs = ['unintended', 'toxic', 'test']
def get_src_id(x):
  return tf.cast(tf.math.equal(x, 'unintended'), tf.int32) \
         + tf.cast(tf.math.equal(x, 'toxic'), tf.int32) * 2 \
         + tf.cast(tf.math.equal(x, 'test'), tf.int32) * 3 \
         - 1