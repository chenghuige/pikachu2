#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:33.472128
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import gamma

import sys 
import os

import tensorflow as tf
from tensorflow.keras import backend as K
import global_objectives
import melt as mt
from .config import *

def tower_loss_fn(y_true, y_pred, x):
  pred = y_pred
  pred = tf.cast(pred, tf.float32)
  margin = 0.1
  pos_score = pred[:, 0]
  neg_score = pred[:, 1]
  loss_ = tf.nn.relu(margin - (pos_score - neg_score))
  if 'weight' in x:
    loss_ *= tf.cast(x['weight'], tf.float32)
  loss = mt.reduce_over(loss_)
  return loss

# https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247505145&idx=1&sn=42467a5475b64d3031a46594261600d2&chksm=96ea0b79a19d826fb7e040968bbe928daf528d38e63d916b05cbc60a5d350755e619abc2aff8#rd
def multilabel_categorical_crossentropy(y_true, y_pred, x):
  """多标签分类的交叉熵
  说明：y_true和y_pred的shape一致，y_true的元素非0即1，
        1表示对应的类为目标类，0表示对应的类为非目标类。
  """

  actions = []
  for action in FLAGS.loss_list:
    actions.append(tf.cast(x[action], tf.float32))

  y_true = tf.concat(actions, -1)

  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  y_pred = (1. - 2. * y_true) * y_pred
  y_pred_neg = y_pred - y_true * 1e12
  y_pred_pos = y_pred - (1. - y_true) * 1e12
  zeros = K.zeros_like(y_pred[..., :1])
  y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
  y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
  neg_loss = tf.math.reduce_logsumexp(y_pred_neg, axis=-1)
  pos_loss = tf.math.reduce_logsumexp(y_pred_pos, axis=-1)
  return mt.reduce_over(neg_loss + pos_loss)

def rdrop_loss(y_true, y_pred, x, model):
  loss_fn_ = get_loss_fn()
  loss1 = loss_fn_(y_true, y_pred, x)
  loss2 = loss_fn_(y_true, model.pred2, x)
  kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
  kl_loss = kl(tf.nn.softmax(y_pred, -1), tf.nn.softmax(model.pred2, -1))
  kl_loss = mt.reduce_over(kl_loss)
  loss = 0.5 * (loss1 + loss2) + 10 * kl_loss
  return loss

def get_loss_fn():  
  if FLAGS.loss_fn == 'softmax':
    return multilabel_categorical_crossentropy

  return loss_fn

def loss_fn(y_true, y_pred, x, model):
  pred = y_pred if not FLAGS.action_loss else model.logit
  pred = tf.cast(pred, tf.float32)
  
  if FLAGS.loss_fn == 'bce':
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.NONE)
  elif FLAGS.loss_fn == 'focal':
    import focal_loss
    loss_func = focal_loss.BinaryFocalLoss(gamma=2., from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  elif FLAGS.loss_fn == 'auc':
    loss_func = lambda x, y: global_objectives.roc_auc_loss(x, y)[0]
  else:
    raise ValueError(FLAGS.loss_fn)

  loss = 0
  loss_list = []
  total_weight = sum(FLAGS.weights[:len(FLAGS.loss_list)])
  # total_weight = len(FLAGS.loss_list)
  for i, action in enumerate(FLAGS.loss_list):
    y_true_ = tf.cast(x[action], tf.float32)
    y_pred_ = tf.expand_dims(pred[:, i], -1)
    if not FLAGS.hack_loss:
      # 走这里
      # (bs, 1), (bs, 1) -> (bs,)
      loss_ = loss_func(y_true_, y_pred_) 
      ##  0.17058875 / 1024 = 0.0006663623
      #f.Tensor(0.17058875, shape=(), dtype=float32)                                                                                                                                                            | 1/1638 [00:08<3:47:19,  8.33s/it, loss=0.1715]
      # tf.Tensor(0.682355, shape=(), dtype=float32)
      # tf.Tensor(0.0006663623, shape=(), dtype=float32)  # before
      # print(mt.reduce_over(loss_))
      # print(mt.reduce_over2(loss_))
      ## (bs, 1), (bs,) -> ()
      # print(mt.reduce_over2(loss_func(tf.cast(x[action], tf.float32), pred[:, i])))
    else:
      ## (bs, 1), (bs,) -> () 会多 / batch_size_per_gpu.. 这个是之前bug的地方 这里只为了复现对比
      loss_ = loss_func(tf.cast(x[action], tf.float32), pred[:, i])

    if 'weight' in x:
      loss_ *= tf.cast(x['weight'], tf.float32)
    if not FLAGS.hack_loss:
      if FLAGS.reduce_loss:
        loss_ = mt.reduce_over(loss_) 
      else:
        loss_ = tf.reduce_sum(loss_)
    else:
      loss_ = mt.reduce_over2(loss_)

    if FLAGS.action_loss:
      mask = tf.cast(x['action'] > 0, tf.float32)
      if action != 'action':
        loss_ = loss_ * mask 
      else:
        loss_ = loss_ * mask + loss_ * total_weight * (1 - mask)

    if FLAGS.uncertain_loss:
      loss_list.append(loss_)
    elif not FLAGS.weight_loss:
      loss += loss_ 
    else:
      loss += (FLAGS.weights[i] / total_weight) * loss_

  if FLAGS.uncertain_loss:
    loss = model.uncertain(loss_list)

  if FLAGS.sample_method:
    aux_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    label = tf.squeeze(tf.zeros_like(x['doc']), -1)
    aux_loss = aux_loss_func(label, model.dot)
    aux_loss = mt.reduce_over(aux_loss)
    loss += aux_loss * FLAGS.aux_loss_rate
    # print(model.dot)
    # print(loss, aux_loss)
    
  # hack for old way, with batch size 4096 total, 2 gpu, lr 0.048 and loss / 2048.
  if FLAGS.hack_loss:
    loss /= 2048.

  return loss

def get_loss(model=None):
  loss_fn_ = model.get_loss()  
  return loss_fn_ 
