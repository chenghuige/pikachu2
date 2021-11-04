#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:43.549005
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 

# def earth_mover_loss(y_true, y_pred):
#   cdf_ytrue = tf.cumsum(y_true, axis=-1)
#   cdf_ypred = tf.cumsum(y_pred, axis=-1)
#   samplewise_emd = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(cdf_ytrue - cdf_ypred)), axis=-1))
#   return samplewise_emd

import tensorflow.keras.backend as K

from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

from config import *
import gezi

def earth_mover_loss(y_true, y_pred):
  cdf_ytrue = K.cumsum(y_true, axis=-1)
  cdf_ypred = K.cumsum(y_pred, axis=-1)
  samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
  return samplewise_emd

# reference link: https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
#.. can no simple load
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.compat.v1.name_scope(name):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true = tf.one_hot(y_true, NUM_CLASSES)
        repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype=tf.float32)
        repeat_op_sq = tf.square((repeat_op - tf.transpose(a=repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype=tf.float32)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(input_tensor=pred_, axis=1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(input_tensor=pred_, axis=1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(input_tensor=pred_norm, axis=0)
        hist_rater_b = tf.reduce_sum(input_tensor=y_true, axis=0)
    
        conf_mat = tf.matmul(tf.transpose(a=pred_norm), y_true)
    
        nom = tf.reduce_sum(input_tensor=weights * conf_mat)
        denom = tf.reduce_sum(input_tensor=weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) / tf.cast(bsize, dtype=tf.float32))
    
        return nom * 0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred) * 0.5

def get_loss(loss_type=None):
  if 'regression' in loss_type:
    if 'sigmoid2' in loss_type:
      return tf.compat.v1.losses.sigmoid_cross_entropy
    if 'mae' not in loss_type:
      return tf.compat.v1.losses.mean_squared_error
    else:
      return tf.compat.v1.losses.absolute_difference
  elif 'ordinal' in loss_type:
    return tf.compat.v1.losses.sigmoid_cross_entropy
  elif 'earth' in loss_type:
    return earth_mover_loss
  else:
    # classification
    if 'kappa' in loss_type:
      return kappa_loss
    # def func_(y, y_):
    #   gezi.sprint(y)
    #   gezi.sprint(y_)
    #   y_ = tf.cast(y_, tf.int64)
    #   return tf.losses.sparse_softmax_cross_entropy(y, y_)
    # return func_
    return tf.compat.v1.losses.sparse_softmax_cross_entropy
    


