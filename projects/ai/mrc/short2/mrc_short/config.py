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

flags.DEFINE_string('transformer', 'bert-base-chinese', '')  
flags.DEFINE_bool('from_pt', False, '')
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_string('pooling', 'concat', '')
flags.DEFINE_bool('mdrop', False, '')
flags.DEFINE_string('dev_pkl', '../input/dev.pkl', '')
flags.DEFINE_string('dev_json', '../input/dev.lat.json', '')

def init():
  FLAGS.static_input = True
