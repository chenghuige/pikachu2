#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2021-01-10 17:16:24.668713
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

CLASSES = ['start', 'end']
NUM_CLASSES = len(CLASSES)

flags.DEFINE_string('transformer', 'bert-base-chinese', 'hfl/chinese-electra-180g-base-discriminator')  
flags.DEFINE_bool('from_pt', False, '')
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_string('pooling', 'concat', '')
flags.DEFINE_bool('mdrop', True, '')
flags.DEFINE_bool('use_mask', True, '')
flags.DEFINE_bool('use_segment', True, '')
flags.DEFINE_bool('use_weight', False, '')

def init():
  FLAGS.static_input = True
