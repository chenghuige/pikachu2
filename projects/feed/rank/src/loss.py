#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2019-07-28 15:15:35.162774
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

from tensorflow import keras
from tensorflow.keras import backend as K

import pickle 
from functools import partial
import gezi
logging = gezi.logging
import melt

from projects.feed.rank.src import util 
import global_objectives

qt_uniform = None
qt_normal = None 

def init():
  global qt_uniform 
  global qt_normal
  try:
    data_dir = FLAGS.data_dir
    qt_uniform = pickle.load(open(f'{data_dir}/qt.uniform.pkl', 'rb'))
    qt_normal = pickle.load(open(f'{data_dir}/qt.normal.pkl', 'rb'))  
  except Exception: 
    logging.debug('could not load scaler, you may need to use [python prepare/gen-scaler.py data_dir]')
  
def uniform_scale(x):
  #print('before scale', x)
  flag = (x > 0).astype(np.float32)
  x = qt_uniform.transform(x.reshape(-1, 1))
  x *= flag 
  #print('after scale', x)
  return x

def normal_scale(x):
  #print('before scale', x)
  flag = (x > 0).astype(np.float32)
  mid = FLAGS.duration_log_max / 2
  x = (qt_normal.transform(x.reshape(-1, 1)) + mid) / FLAGS.duration_log_max
  for i in range(len(x)):
    x[i] = max(x[i], 0.1)
    x[i] = min(x[i], 1.)
  x *= flag
  #print('after scale', x)
  return x

def duration2class(x):
  if x == 0:
    return 0
  elif x < 3:
    return 1 
  elif x < 4:
    return 2
  elif x < 5:
    return 3
  elif x < 6:
    return 4
  else:
    return 5

# TODO duration t0 255 bins
def duration2class2(x):
  if x == 0:
    return 0
  if x >= 300:
    return 299
  return int(x) 

def duration2classes(x):
  result = np.asarray(list(map(duration2class, x)))
  return result

def duration2classes2(x):
  result = np.asarray(list(map(duration2class2, x)))
  return result

# 这里参数没有weights 但是内部通过x取得weights另外一种方法是参数直接有weights melt的训练里面会传递
# 注意model.get_loss 如果传递self.input_实际为None 未被初始化 train + evaluated模式报错信息是比较诡异的 
# TypeError: An op outside of the function building code is being passed
# a "Graph" tensor. It is possible to have Graph tensors
# leak out of the function building context by including a
# tf.init_scope in your function building code.
# For example, the following function will fail:
#   @tf.function
#   def has_init_scope():
#     my_constant = tf.constant(1.)
#     with tf.init_scope():
#       added = my_constant * 2
# The graph tensor has name: IteratorGetNext:12
# 单独evaluate模式会报错x是None 本质还是构图有问题

def multi_obj_duration_loss(y, y_, x, model, uncertain=None):  
  duration = x['duration']
  duration = tf.cast(duration, tf.float32)
  dur_unknown = tf.cast(duration < 0, tf.float32)
  duration = tf.math.minimum(duration, float(FLAGS.max_duration))
  #if FLAGS.multi_obj_duration_loss == 'mean_squared_error':
  if not FLAGS.duration_weight_obj_nolog:
    duration = tf.math.log(duration + 1.)

  if FLAGS.finish_ratio_as_dur:
    dur_prob = util.get_finish_ratio(x, max_video_time=FLAGS.max_duration)
  else:
    if FLAGS.duration_scale == 'log':
      dur_prob = tf.minimum(duration / FLAGS.duration_log_max, 1.)
    elif FLAGS.duration_scale == 'minmax':
      dur_prob = duration / float(FLAGS.max_duration)
    elif FLAGS.duration_scale == 'uniform':
      #dur_prob = tf.squeeze(tf.numpy_function(scale, [tf.expand_dims(duration, -1)], tf.float32), -1)
      dur_prob = tf.numpy_function(uniform_scale, [duration], tf.float32)
    elif FLAGS.duration_scale == 'normal':
      dur_prob = tf.numpy_function(normal_scale, [duration], tf.float32)
    elif FLAGS.duration_scale == 'sigmoid':
      dur_prob = tf.math.sigmoid(duration)
    else:
      dur_prob = duration

  if FLAGS.finish_ratio_as_click:
    y = util.get_finish_ratio(x, max_video_time=FLAGS.max_duration)

  ratio = FLAGS.multi_obj_duration_ratio 
  ratio2 = FLAGS.multi_obj_duration_ratio2

  if FLAGS.use_deep_position_emb:
    model.y_click = melt.prob2logit(tf.math.sigmoid(model.y_click) * tf.math.sigmoid(model.y_pos))

  kwargs = {}
  if FLAGS.use_weight:
    kwargs['weights'] = x['weight']
  if not FLAGS.rank_loss:
    kwargs['reduction'] = 'none'
    click_loss = tf.compat.v1.losses.sigmoid_cross_entropy(y, model.y_click, **kwargs)
  else:
    # click_loss = global_objectives.roc_auc_loss(y, model.y_click, **kwargs)[0]
    click_loss = global_objectives.precision_recall_auc_loss(y, model.y_click, reuse=tf.AUTO_REUSE, scope='click_loss', **kwargs)[0]
  dur_loss_fn = getattr(tf.compat.v1.losses, FLAGS.multi_obj_duration_loss)
  if model.dur_need_sigmoid:
    y_dur = model.y_dur if FLAGS.use_jump_loss else model.logit
  else:
    y_dur = model.prob_dur if FLAGS.use_jump_loss else model.prob
  if not FLAGS.rank_dur_loss:
    kwargs['reduction'] = 'none'
    dur_loss = dur_loss_fn(dur_prob, y_dur, **kwargs) 
  else:
    # dur_loss = global_objectives.roc_auc_loss(dur_prob, y_dur, **kwargs)[0] 
    dur_loss = global_objectives.precision_recall_auc_loss(dur_prob, y_dur, reuse=tf.AUTO_REUSE, scope='dur_loss', **kwargs)[0] 
  dur_loss *= (1. - dur_unknown)
  if FLAGS.use_jump_loss:
    dur_loss *= y

  if uncertain is None:
    if FLAGS.multi_obj_sum_loss:
      loss = click_loss + dur_loss
    else:
      loss = click_loss * (1. - ratio) + dur_loss * ratio 
  else:
    loss = uncertain([click_loss, dur_loss])

  # tf.print()
  # tf.print(tf.reduce_mean(click_loss), tf.reduce_mean(dur_loss), tf.reduce_mean(loss))

  if FLAGS.finish_loss:
    finish_ratio = util.get_finish_ratio(x, max_video_time=FLAGS.max_duration)
    loss = loss * (1. - ratio2) + tf.compat.v1.losses.sigmoid_cross_entropy(finish_ratio, model.y_finish, **kwargs) * ratio2 * (1. - dur_unknown)

  if FLAGS.use_deep_position_emb:
    loss = loss + 0.1 * tf.compat.v1.losses.sigmoid_cross_entropy(y, model.y_pos, **kwargs)

  if FLAGS.use_weight:
    loss = tf.reduce_sum(loss) / tf.reduce_sum(x['weight'])
  else:
    loss = tf.reduce_mean(loss)

  # if not tf.executing_eagerly():
  #   tag = 'train' if K.learning_phase() else 'valid'
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/all', loss)
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/click', tf.reduce_sum(click_loss) / tf.reduce_sum(x['weight']))
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/dur', tf.reduce_sum(dur_loss * (1. - dur_unknown)) / tf.reduce_sum(x['weight']))
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/dur_jump', tf.reduce_sum(dur_loss * (1. - dur_unknown) * y) / tf.reduce_sum(x['weight']))
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/weight/mean', tf.reduce_mean(x['weight']))
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/weight/max', tf.reduce_max(x['weight']))
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/weight/min', tf.reduce_min(x['weight']))
  #   tf.compat.v1.summary.scalar(f'stats/{tag}/feat_len', melt.get_shape(x['index'], 1))
  #   tf.compat.v1.summary.scalar(f'stats/{tag}/pred_click', tf.reduce_mean(model.prob_click))
  #   tf.compat.v1.summary.scalar(f'stats/{tag}/pred_dur', tf.reduce_mean(model.prob_dur))
  #   tf.compat.v1.summary.scalar(f'stats/{tag}/pred', tf.reduce_mean(model.prob))
    
  return loss

def finish_loss(y, y_, x, model):
  # print(list(zip(x['video_time_str'].numpy(), x['video_time'].numpy())))
  video_time = tf.math.minimum(x['video_time'], FLAGS.max_duration)
  video_time = tf.math.maximum(video_time, FLAGS.min_video_time)
  finish_ratio = tf.cast(x['duration'], tf.float32) / tf.cast(video_time, tf.float32)
  finish_ratio = tf.math.minimum(finish_ratio, 1.)
  finish_loss_ = tf.compat.v1.losses.sigmoid_cross_entropy(finish_ratio, y_)
  ratio = FLAGS.finish_loss_ratio
  loss = multi_obj_duration_loss(y, y_, x, model)
  loss = finish_loss_ * ratio + loss * (1 - ratio)

  return loss

def sigmoid_cross_entropy(y, y_, x, model):
  weights = tf.reshape(x['weight'], tf.shape(y))
  return tf.compat.v1.losses.sigmoid_cross_entropy(y, model.logit, weights=weights)

class MultiLoss(object):
  def __init__(self, model, uncertain=None):
    self.model = model
    self.uncertain = uncertain
    self.__name__ = "MultiLoss"
  
  def __call__(self, y_true, y_pred, sample_weight=None):
    return multi_obj_duration_loss(y_true, y_pred, None, self.model, self.uncertain)

def get_loss_fn():
  if FLAGS.multi_obj_type is not None:
    if FLAGS.multi_obj_uncertainty_loss:
      uncertain = melt.layers.UncertaintyLoss()
      return partial(multi_obj_duration_loss, uncertain=uncertain)
    return multi_obj_duration_loss
  elif FLAGS.finish_loss:
    return finish_loss
  else:
    return sigmoid_cross_entropy

