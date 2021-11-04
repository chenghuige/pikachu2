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

CLASSES = ['yes', 'no', 'depences', 'other', 'unk']
NUM_CLASSES = len(CLASSES)

flags.DEFINE_string('transformer', 'bert-base-chinese', '')  
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_bool('use_all2', False, '')
flags.DEFINE_string('pooling', 'concat', '')
flags.DEFINE_bool('mdrop', False, '')

def init():
  if 'pad' in FLAGS.input:
    FLAGS.static_input = True
