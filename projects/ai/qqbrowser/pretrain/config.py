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

flags.DEFINE_string('transformer', 'hfl/chinese-bert-wwm-ext', 'hfl/chinese-bert-wwm-ext  bert-base-chinese')  
flags.DEFINE_string('records_name', 'tfrecords', '')
flags.DEFINE_string('records_version', '', '')
flags.DEFINE_alias('rv', 'records_version')
flags.DEFINE_bool('custom_model', False, '')
flags.DEFINE_string('embedding_path', None, '')
flags.DEFINE_integer('vocab_size', None, '451681 for word')
flags.DEFINE_integer('hidden_size', None, '')
flags.DEFINE_integer('num_attention_heads', None, '')
flags.DEFINE_bool('word', False, '')
flags.DEFINE_bool('use_vision', False, '')
flags.DEFINE_integer('max_frames', 32, '')
flags.DEFINE_integer('frame_embedding_size', 1536, '')


import gezi
from gezi import logging
import melt as mt
 
def init():
  FLAGS.records_name += FLAGS.records_version
  FLAGS.train_files = [
    *gezi.list_files(f'../input/{FLAGS.records_name}/train/*.tfrec'),
    *gezi.list_files(f'../input/{FLAGS.records_name}/valid/*.tfrec'),
    *gezi.list_files(f'../input/{FLAGS.records_name}/test_a/*.tfrec'),
    *gezi.list_files(f'../input/{FLAGS.records_name}/test_b/*.tfrec'),
  ]
  # FLAGS.valid_files = gezi.list_files('../input/tfrecords2/test_a/*.tfrec')
  FLAGS.static_input = True
