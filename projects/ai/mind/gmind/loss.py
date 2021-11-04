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
import melt as mt
from tensorflow.keras import backend as K
from .config import *
import global_objectives

def loss_fn(y_true, y_pred, x, model):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  if FLAGS.lm_target:
    return mt.losses.sampled_bilm_loss(y_true, model.his_embs, model.softmax_loss_function)

  if FLAGS.label_smoothing_rate:
    prob = tf.random.uniform([], minval=0., maxval=FLAGS.label_smoothing_rate)
    cond = tf.cast(tf.not_equal(y_true, 1), tf.float32)
    y_true =  cond * prob + (1. - cond)
  
  kwargs = {'reduction': 'none'}

  bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  if FLAGS.loss_type == 'sigmoid':
    loss = bce(y_true, y_pred) 
  elif FLAGS.loss_type == 'roc_auc':
    loss = global_objectives.roc_auc_loss(y_true, y_pred, scope='roc_auc_loss', **kwargs)[0]
  elif FLAGS.loss_type == 'pr_auc':
    loss = global_objectives.precision_recall_auc_loss(y_true, y_pred, reuse=tf.AUTO_REUSE, scope='pr_auc_loss', **kwargs)[0]
  elif FLAGS.loss_type == 'focal':
    loss = mt.losses.focal_loss(y_true, y_pred)
  elif FLAGS.loss_type == 'pair':
    point_loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred, **kwargs) 
    perm = tf.random.shuffle(tf.range(tf.shape(y_true)[0]))
    y_true2 = tf.gather(y_true, perm, axis=0)
    y_pred2 = tf.gather(y_pred, perm, axis=0)
    impression_id = x['impression_id']
    impression_id2 = tf.gather(impression_id, perm, axis=0)
    point_loss2 = tf.gather(point_loss, perm, axis=0)
    mask = tf.math.abs(y_true - y_true2) * tf.cast(K.equal(impression_id, impression_id2), tf.float32)
    margin = 0.1
    pair_loss = mask * tf.nn.relu(0.1 - ((2. * y_true - 1.) * y_pred + (1. - 2. * y_true2) * y_pred2))
    point_loss *= 10.
    loss = point_loss + pair_loss
  elif FLAGS.loss_type == 'pair2':
    point_loss = bce(y_true, y_pred)
    pair_loss = 0.
    # y_prob = tf.nn.sigmoid(y_pred)
    y_prob = y_pred
    impression_id = x['impression_id']
    margin = 0.1
    for _ in range(FLAGS.num_pairs):
      perm = tf.random.shuffle(tf.range(tf.shape(y_true)[0]))
      y_true2 = tf.gather(y_true, perm, axis=0)
      y_prob2 = tf.gather(y_prob, perm, axis=0)
      impression_id2 = tf.gather(impression_id, perm, axis=0)
      point_loss2 = tf.gather(point_loss, perm, axis=0)
      mask = tf.math.abs(y_true - y_true2) * tf.cast(K.equal(impression_id, impression_id2), tf.float32)
      pair_loss += mask * tf.nn.relu(margin - ((2. * y_true - 1.) * y_prob + (1. - 2. * y_true2) * y_prob2))
      # point_loss *= 10.
    loss = point_loss + pair_loss
  else:
    raise ValueError(FLAGS.loss_type)

  use_weight = 'weight' in x and K.learning_phase()
  if use_weight:
    weights = tf.cast(x['weight'], tf.float32)
    loss *= weights
    
  loss = mt.reduce_over(loss)

  main_loss = loss
  if FLAGS.aux_loss_rate:
    aux_loss = mt.losses.sampled_bilm_loss(x['history'], model.his_embs, model)
    loss = loss + aux_loss * FLAGS.aux_loss_rate
 
  # if not tf.executing_eagerly():
  #   tag = 'train' if K.learning_phase() else 'valid'
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/main', main_loss)
  #   tf.compat.v1.summary.scalar(f'info/{tag}/his_max', tf.reduce_max(x['hist_len']))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/his_min', tf.reduce_min(x['hist_len']))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/his_mean', tf.reduce_mean(x['hist_len']))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/impression_max', tf.reduce_max(x['impression_len']))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/impression_min', tf.reduce_min(x['impression_len']))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/impression_mean', tf.reduce_mean(x['impression_len']))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/click_ratio', tf.reduce_mean(tf.cast(y_true, tf.float32)))
  #   tf.compat.v1.summary.scalar(f'info/{tag}/pred_mean', tf.reduce_mean(tf.sigmoid(y_pred)))
  #   if use_weight:
  #     tf.compat.v1.summary.scalar(f'info/{tag}/weight_mean', tf.reduce_mean(x['weight']))
  #   if FLAGS.aux_loss_rate or FLAGS.lm_target:
  #     tf.compat.v1.summary.scalar(f'loss/{tag}/aux', aux_loss)

  #   if 'pair' in FLAGS.loss_type:
  #     tf.compat.v1.summary.scalar(f'loss/{tag}/point', tf.reduce_mean(point_loss))
  #     tf.compat.v1.summary.scalar(f'loss/{tag}/pair', tf.reduce_mean(pair_loss))

  return loss

def get_loss(model=None):
  if not FLAGS.custom_loss:
    loss_fn_ = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  else:
    loss_fn_ = model.get_loss()  
  return loss_fn_
  