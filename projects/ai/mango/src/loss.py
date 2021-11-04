#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2020-05-24 09:17:09.674068
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
import gezi
logging = gezi.logging
import melt
from tensorflow.keras import backend as K
from projects.ai.mango.src.config import *

def loss_fn(y_true, y_pred, x, model):
  if FLAGS.lm_target:
    return melt.losses.sampled_bilm_loss(y_true, model.his_embs, model.softmax_loss_function)

  if FLAGS.label_smoothing_rate:
    prob = tf.random.uniform([], minval=0., maxval=FLAGS.label_smoothing_rate)
    cond = tf.cast(tf.not_equal(y_true, 1), tf.float32)
    y_true =  cond * prob + (1. - cond)
  
  kwargs = {}
  use_weight = FLAGS.use_weight and K.learning_phase()
  if use_weight:
    kwargs['reduction'] = 'none'
  loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred, **kwargs)   
  if use_weight:
    loss = tf.reduce_sum(loss) / tf.reduce_sum(x['weight'])

  main_loss = loss
  if FLAGS.aux_loss_rate:
    aux_loss = melt.losses.sampled_bilm_loss(x['watch_vids'], model.his_embs, model)
    loss = loss + aux_loss * FLAGS.aux_loss_rate
 
  if not tf.executing_eagerly():
    tag = 'train' if K.learning_phase() else 'valid'
    tf.compat.v1.summary.scalar(f'loss/{tag}/main', main_loss)
    if FLAGS.aux_loss_rate or FLAGS.lm_target:
      tf.compat.v1.summary.scalar(f'loss/{tag}/aux', aux_loss)
    tf.compat.v1.summary.scalar(f'stats/{tag}/watch_vids/max', tf.reduce_max(melt.length(x['watch_vids'])))
    tf.compat.v1.summary.scalar(f'stats/{tag}/watch_vids/min', tf.reduce_min(melt.length(x['watch_vids'])))
    tf.compat.v1.summary.scalar(f'stats/{tag}/watch_vids/mean', tf.reduce_mean(melt.length(x['watch_vids'])))

  return loss
