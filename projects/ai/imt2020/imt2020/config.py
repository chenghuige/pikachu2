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

CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
INPUT_DIM = CHANNEL_SHAPE_DIM1 * CHANNEL_SHAPE_DIM2 * CHANNEL_SHAPE_DIM3

flags.DEFINE_integer('NUM_FEEDBACK_BITS', 768, '')
flags.DEFINE_bool('from_pt', False, '')
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_string('pooling', 'concat', '')
flags.DEFINE_bool('mdrop', False, '')

def init():
  FLAGS.static_input = True
