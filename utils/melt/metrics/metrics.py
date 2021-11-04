#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics.py
#        \author   chenghuige  
#          \date   2020-10-02 14:51:13.713751
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import confusion_matrix

import gezi

# TODO CategoryMIoU 多标签问题对每个样本计算类别MIoU
class CategoryMIoU(tf.keras.metrics.Metric):
  def __init__(self, name='CatMIoU', **kwargs):
    super(CategoryMIoU,self).__init__(name=name, **kwargs) 
    self.total = tf.constant(0., tf.float32)
    self.count = tf.constant(0, tf.int32)
      
  def reset_states(self):
    self.total = 0.
    self.count = 0
          
  def update_state(self, y_true, y_pred,sample_weight=None):
    if K.learning_phase():
      return
    self.count += 1
    y_pred = tf.cast(y_pred > 0, y_true.dtype)
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1) - intersection
    self.total += tf.cast(tf.reduce_mean(intersection / union), tf.float32)
      
  def result(self):
    return self.total / tf.cast(self.count, tf.float32)

class Accuracy(tf.keras.metrics.Accuracy):
  def __init__(self, name='ACC', dtype=None):
    super(Accuracy, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    if K.learning_phase():
      return
    return super().update_state(y_true, y_pred, sample_weight=sample_weight)

class SparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  def __init__(self, name='ACC', dtype=None):
    super(SparseCategoricalAccuracy, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    if K.learning_phase():
      return
    return super().update_state(y_true, y_pred, sample_weight=sample_weight)

class AUC(tf.keras.metrics.AUC):
  def __init__(self, name='AUC', dtype=None):
    super(AUC, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    if K.learning_phase():
      return
    return super().update_state(y_true, y_pred, sample_weight=sample_weight) 

class MeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self, num_classes, name='MIoU', dtype=None):
    super(MeanIoU, self).__init__(num_classes, name=name, dtype=dtype)

    self.total_cm = self.add_weight(
          'total_confusion_matrix',
          shape=(num_classes, num_classes),
          initializer=init_ops.zeros_initializer,
          dtype=dtypes.int32)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # TODO HACK 训练中禁止更新 对应tqdm_progress_bar也修改 （因为当前无法做到一定train step做一次IoU计算以及 每次自动reset state 还有就是valid和train混合更新相同IoU问题）
    if K.learning_phase():
      return

    # If not onehot assume (bs, x, x, 1) else (bs, x, x, num_classes)
    if y_true.shape == y_pred.shape:
      y_true = tf.argmax(y_true, axis=-1)

    # assume (bs, x, x, num_classes)
    if not gezi.get('CLASS_WEIGHTS'):
      y_pred = tf.argmax(y_pred, axis=-1)
    else:
      y_pred = tf.argmax(tf.math.softmax(y_pred, -1) * tf.constant(gezi.get('CLASS_WEIGHTS'), tf.float32), axis=-1)

    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)

    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
      y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = array_ops.reshape(y_true, [-1])

    if sample_weight is not None:
      sample_weight = math_ops.cast(sample_weight, self._dtype)
      if sample_weight.shape.ndims > 1:
        sample_weight = array_ops.reshape(sample_weight, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        y_true,
        y_pred,
        self.num_classes,
        weights=sample_weight,
        dtype=dtypes.int32)  # chg change from float64 to int2 for tpu
    return self.total_cm.assign_add(current_cm)

class FWIoU(MeanIoU):
  def __init__(self, num_classes, name='FWIoU', dtype=None):
    super(FWIoU, self).__init__(num_classes, name=name, dtype=dtype)

  def result(self):
    sum_all = math_ops.cast(
        math_ops.reduce_sum(self.total_cm), dtype=self._dtype)
    sum_over_row = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = math_ops.cast(
        array_ops.diag_part(self.total_cm), dtype=self._dtype)

    freq_weights = math_ops.div_no_nan(sum_over_col, sum_all)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    iou = math_ops.div_no_nan(true_positives, denominator)

    fwiou = iou * freq_weights

    fwiou = math_ops.reduce_sum(fwiou, name='fw_iou')
    return fwiou

class SemanticSeg(MeanIoU):
  def __init__(self, num_classes, name='SemanticSeg', dtype=None):
    super(SemanticSeg, self).__init__(num_classes, name=name, dtype=dtype)
    self.num_results = 4
    self.results = None
    self.metric_names = ['FWIoU', 'MIoU', 'ACC/pixel', 'ACC/class'] 

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(y_true, y_pred, sample_weight=sample_weight)

  def result(self):
    sum_all = math_ops.cast(
        math_ops.reduce_sum(self.total_cm), dtype=self._dtype)
    sum_over_row = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = math_ops.cast(
        array_ops.diag_part(self.total_cm), dtype=self._dtype)

    freq_weights = math_ops.div_no_nan(sum_over_col, sum_all)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    iou = math_ops.div_no_nan(true_positives, denominator)

    fwiou = iou * freq_weights
    fwiou = math_ops.reduce_sum(fwiou, name='fw_iou')

    num_valid_entries = math_ops.reduce_sum(
        math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

    iou = math_ops.div_no_nan(math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    sum_trues = math_ops.reduce_sum(true_positives)

    acc_pixel = math_ops.div_no_nan(sum_trues, sum_all)

    acc_class = math_ops.reduce_mean(math_ops.div_no_nan(true_positives, sum_over_col))

    self.results = fwiou, iou, acc_pixel, acc_class 
    return self.results

  def get_cm(self):
    return self.total_cm

  def __len__(self):
    return self.num_results

  def get_metrics(self):
    return get_wrapper_metrics(self)

class WrapperMetrics(tf.keras.metrics.Metric):
  def __init__(self, metric, name, index, **kwargs):
    super(WrapperMetrics,self).__init__(name=name, **kwargs) 
    self.metric = metric
    self.index = index
      
  def reset_states(self):
    if self.index == 0:
      self.metric.reset_states()
          
  def update_state(self, y_true, y_pred,sample_weight=None):
    if self.index == 0:
      self.metric.update_state(y_true, y_pred,sample_weight)
      
  def result(self):
    if self.index == 0:
      self.metric.result()
    return self.metric.results[self.index]

  def get_cm(self):
    return self.metric.get_cm()
  
class MeanMetric(tf.keras.metrics.Metric):
  def __init__(self, name, **kwargs):
    super().__init__(name=name, **kwargs)
    self.val = 0.
    self.steps = 0
    
  def reset_states(self):
    self.val = 0.
    self.steps = 0
    
  def update_state(self, val):
    self.val += val  
    self.steps += 1
    
  def result(self):
    return self.val / self.steps
  
class CurrentMetric(tf.keras.metrics.Metric):
  def __init__(self, name, **kwargs):
    super().__init__(name=name, **kwargs)
    self.val = 0.
    
  def reset_states(self):
    self.val = 0.
    
  def update_state(self, val):
    self.val = val  
    
  def result(self):
    return self.val 

def get_wrapper_metrics(metric):
  num_metrics = len(metric)
  metric_names = metric.metric_names
  return [WrapperMetrics(metric, metric_names[i], i) for i in range(num_metrics)]

def get_segmantic_seg_metrics(num_classes):
  metric = SemanticSeg(num_classes)
  return get_wrapper_metrics(metric)
