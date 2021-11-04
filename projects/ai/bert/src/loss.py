#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2020-04-12 20:31:58.898544
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from absl import flags
FLAGS = flags.FLAGS
import tensorflow as tf


def calc_loss(y_true, y_pred):
  pass

def hinge_loss(y_true, y_pred):
  pos_score = y_pred['pos']
  neg_score = y_pred['neg']
  return melt.losses.hinge(pos_score, neg_score)

def get_loss_fn():
  # return tf.compat.v1.losses.sigmoid_cross_entropy
  if FLAGS.task == 'toxic':
    return tf.keras.losses.BinaryCrossentropy()
  elif FLAGS.task == 'lang':
    return tf.keras.losses.CategoricalCrossentropy()
  elif FLAGS.task == 'translate':
    return hinge_loss
  