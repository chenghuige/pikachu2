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
# from config import *
from projects.ai.ad.src.config import *

def loss_fn(y_true, y_pred, x, model):
  gender = x['gender']
  gender_loss = tf.compat.v1.losses.sigmoid_cross_entropy(gender, model.gender)  
  age = x['age']
  # if 'Cls' in FLAGS.model:
  age_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(age, model.age)
  # else:
  #   age_loss = tf.compat.v1.losses.sigmoid_cross_entropy(age, model.age)

  loss = gender_loss + age_loss

  # if not tf.executing_eagerly():
  #   tag = 'train' if K.learning_phase() else 'valid'
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/all', loss)
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/gender', gender_loss)
  #   tf.compat.v1.summary.scalar(f'loss/{tag}/age', age_loss)

  #   tf.compat.v1.summary.scalar(f'stats/{tag}/seq_len/max', tf.reduce_max(melt.length(x['ad_ids'])))
  #   tf.compat.v1.summary.scalar(f'stats/{tag}/seq_len/min', tf.reduce_min(melt.length(x['ad_ids'])))
  #   tf.compat.v1.summary.scalar(f'stats/{tag}/seq_len/mean', tf.reduce_mean(melt.length(x['ad_ids'])))

  return loss

def loss_age(y_true, y_pred, x, model):
  gender = x['gender']
  gender_loss = tf.compat.v1.losses.sigmoid_cross_entropy(gender, model.gender)  
  age = x['age']
  if 'Cls' in FLAGS.model:
    age_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(age, model.age)
  else:
    age_loss = tf.compat.v1.losses.sigmoid_cross_entropy(age, model.age)

  # loss = gender_loss + age_loss
  loss = age_loss

  if not tf.executing_eagerly():
    tag = 'train' if K.learning_phase() else 'valid'
    tf.compat.v1.summary.scalar(f'loss/{tag}/all', loss)
    tf.compat.v1.summary.scalar(f'loss/{tag}/gender', gender_loss)
    tf.compat.v1.summary.scalar(f'loss/{tag}/age', age_loss)

  return loss

def loss_gender(y_true, y_pred, x, model):
  gender = x['gender']
  gender_loss = tf.compat.v1.losses.sigmoid_cross_entropy(gender, model.gender)  
  age = x['age']
  age_loss = tf.compat.v1.losses.sigmoid_cross_entropy(age, model.age)

  # loss = gender_loss + age_loss
  loss = gender_loss

  if not tf.executing_eagerly():
    tag = 'train' if K.learning_phase() else 'valid'
    tf.compat.v1.summary.scalar(f'loss/{tag}/all', loss)
    tf.compat.v1.summary.scalar(f'loss/{tag}/gender', gender_loss)
    tf.compat.v1.summary.scalar(f'loss/{tag}/age', age_loss)

  return loss

def loss_mse(y_true, y_pred, x, model):
  gender = x['gender']
  gender_loss = tf.compat.v1.losses.sigmoid_cross_entropy(gender, model.gender)  
  age = x['age']
  age_loss = tf.compat.v1.losses.mean_squared_error(age, model.pred_age)

  loss = gender_loss + age_loss

  if not tf.executing_eagerly():
    tag = 'train' if K.learning_phase() else 'valid'
    tf.compat.v1.summary.scalar(f'loss/{tag}/all', loss)
    tf.compat.v1.summary.scalar(f'loss/{tag}/gender', gender_loss)
    tf.compat.v1.summary.scalar(f'loss/{tag}/age', age_loss)

  return loss


def loss_fn_keras(y_true, y_pred):
  x = gezi.get('input')
  model = gezi.get('model')
  gender = x['gender']
  gender_loss = tf.compat.v1.losses.sigmoid_cross_entropy(gender, model.gender)  
  age = x['age']
  age_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(age, model.age)

  return gender_loss + age_loss  

def loss_age_keras(y_true, y_pred):
  # print(y_true, y_pred)
  y_true = tf.cast(y_true, tf.int64)
  age_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(y_true, y_pred)

  return age_loss  
