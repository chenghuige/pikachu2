#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2020-04-15 10:31:47.629732
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

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_integer('emb_size', 32, '')
flags.DEFINE_integer('hidden_size', 32, '')
flags.DEFINE_string('pooling', 'sum', '')
flags.DEFINE_string('encoder', 'GRU', '')
flags.DEFINE_integer('max_len', 5000, '')
flags.DEFINE_float('dropout', 0.3, '')
flags.DEFINE_float('rdropout', 0.3, '')

flags.DEFINE_float('gender_thre', 0.5, '')

flags.DEFINE_integer('num_layers', 1, '')
flags.DEFINE_bool('concat_layers', True, '')

flags.DEFINE_bool('self_match', False, '')

flags.DEFINE_integer('num_heads', 2, '')

flags.DEFINE_string('lm_target', None, '')

flags.DEFINE_integer('vocab_size', 4027362, 'min count 5 110w, min count 10 60w, total cid 3412774, total ad id 3027362')

flags.DEFINE_bool('train_emb', False, '')

flags.DEFINE_bool('use_mask', True, '')

flags.DEFINE_bool('use_w2v', False, '')
