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
from projects.ai.mind.src.config import *
import global_objectives

def loss_fn(y_true, y_pred, x, model):
  if FLAGS.lm_target:
    return melt.losses.sampled_bilm_loss(y_true, model.his_embs, model.softmax_loss_function)

  if FLAGS.label_smoothing_rate:
    prob = tf.random.uniform([], minval=0., maxval=FLAGS.label_smoothing_rate)
    cond = tf.cast(tf.not_equal(y_true, 1), tf.float32)
    y_true =  cond * prob + (1. - cond)
  
  kwargs = {}
  use_weight = FLAGS.use_weight and K.learning_phase()

  if FLAGS.loss_type == 'sigmoid':
    kwargs['reduction'] = 'none'
    loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred, **kwargs)   
  elif FLAGS.loss_type == 'roc_auc':
    loss = global_objectives.roc_auc_loss(y_true, y_pred, scope='roc_auc_loss', **kwargs)[0]
  elif FLAGS.loss_type == 'pr_auc':
    loss = global_objectives.precision_recall_auc_loss(y_true, y_pred, reuse=tf.AUTO_REUSE, scope='pr_auc_loss', **kwargs)[0]
  else:
    raise ValueError(FLAGS.loss_type)

  if use_weight:
    loss = tf.reduce_sum(loss) / tf.reduce_sum(x['weight'])
  else:
    loss = tf.reduce_mean(loss)

  main_loss = loss
  if FLAGS.aux_loss_rate:
    aux_loss = melt.losses.sampled_bilm_loss(x['history'], model.his_embs, model)
    loss = loss + aux_loss * FLAGS.aux_loss_rate
 
  if not tf.executing_eagerly():
    tag = 'train' if K.learning_phase() else 'valid'
    tf.compat.v1.summary.scalar(f'loss/{tag}/main', main_loss)
    tf.compat.v1.summary.scalar(f'info/{tag}/his_max', tf.reduce_max(model.history_len))
    tf.compat.v1.summary.scalar(f'info/{tag}/his_min', tf.reduce_min(model.history_len))
    tf.compat.v1.summary.scalar(f'info/{tag}/his_mean', tf.reduce_mean(model.history_len))
    tf.compat.v1.summary.scalar(f'info/{tag}/impression_max', tf.reduce_max(model.impression_len))
    tf.compat.v1.summary.scalar(f'info/{tag}/impression_min', tf.reduce_min(model.impression_len))
    tf.compat.v1.summary.scalar(f'info/{tag}/impression_mean', tf.reduce_mean(model.impression_len))
    tf.compat.v1.summary.scalar(f'info/{tag}/click_ratio', tf.reduce_mean(tf.cast(y_true, tf.float32)))
    tf.compat.v1.summary.scalar(f'info/{tag}/pred_mean', tf.reduce_mean(tf.sigmoid(y_pred)))
    if use_weight:
      tf.compat.v1.summary.scalar(f'info/{tag}/weight_mean', tf.reduce_mean(x['weight']))
    if FLAGS.aux_loss_rate or FLAGS.lm_target:
      tf.compat.v1.summary.scalar(f'loss/{tag}/aux', aux_loss)

  return loss
