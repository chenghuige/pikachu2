#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2021-07-31 09:00:06.027197
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import gezi
from gezi import logging
import melt as mt
 
flags.DEFINE_integer('window_size', 8, '')
flags.DEFINE_integer('emb_dim', 128, '')
flags.DEFINE_integer('num_negs', 4, '')
flags.DEFINE_string('day', '14.5', '')
flags.DEFINE_string('attr', 'doc', '')
flags.DEFINE_string('records_name', 'tfrecords/w2v', '')
flags.DEFINE_string('sample_method', 'log_uniform', '')
flags.DEFINE_integer('start_id', 2, '')

vocabs = {}

def init():
  vocabs[FLAGS.attr] = gezi.Vocab(f'../input/{FLAGS.attr}_vocab.txt')
  FLAGS.input = f'../input/{FLAGS.records_name}/{FLAGS.day}/{FLAGS.attr}/{FLAGS.window_size}/*.tfrec'
  FLAGS.static_input = True

  if FLAGS.sample_method == 'log_uniform':
    FLAGS.batch_parse = False
  elif FLAGS.sample_method == 'batch':
    pass
  else:
    raise ValueError(FLAGS.sample_method)
