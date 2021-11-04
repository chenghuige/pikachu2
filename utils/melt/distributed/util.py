#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2019-09-23 08:46:06.156701
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

import sys 
import os
import gezi

from official.utils.misc import distribution_utils

class DummyDistributeStrategy(object):
  def __init__(self, **kwargs):
    self.num_replicas_in_sync = 1

  def scope(self):
    return gezi.DummyContextManager()

  def run(self, fn, args=(), kwargs=None, options=None):
    return fn(*args)

  def reduce(self, reduce_op, value, axis):
    return value

  def experimental_distribute_dataset(self, dataset):
    return dataset

gstrategy = None

def reset_strategy():
  global gstrategy
  gstrategy = None

def set_strategy(distribution_strategy=None, num_gpus=None):
  if tf.__version__ < '2':
    return DummyDistributeStrategy()
  global gstrategy
  # TODO 当前这种就不允许修改strategy了, 需要显示 melt.distributed.reset_strategy()
  if gstrategy is None:
    if distribution_strategy is not None and not isinstance(distribution_strategy, str):
      gstrategy = distribution_strategy
    else:
      if distribution_strategy is None and (not num_gpus or num_gpus == 1):
        gstrategy = DummyDistributeStrategy()
        # gstrategy = tf.distribute.MirroredStrategy()
      else:
        gstrategy = distribution_utils.get_distribution_strategy(
            distribution_strategy=distribution_strategy,
            num_gpus=num_gpus,
            tpu_address=FLAGS.tpu_zone or "")
  return gstrategy

def get_strategy(distribution_strategy=None, num_gpus=None):
  global gstrategy
  if tf.__version__ < '2':
    return DummyDistributeStrategy()
  if not gstrategy:
    gstrategy = set_strategy(distribution_strategy, num_gpus)
  return gstrategy

def has_strategy(strategy=None):
  if not strategy:
    strategy = gstrategy
  return strategy and not isinstance(strategy, DummyDistributeStrategy)

def is_dummy_strategy(strategy=None):
  if not strategy:
    strategy = gstrategy
  return strategy is None or isinstance(strategy, DummyDistributeStrategy)

def get_strategy_scope(strategy):
  if strategy:
    strategy_scope = strategy.scope()
  else:
    strategy_scope = gezi.DummyContextManager()

  return strategy_scope

def concat_merge(x):
  strategy = get_strategy()
  if strategy.num_replicas_in_sync > 1: 
    if isinstance(x, dict):
      for key in x:
        x[key] = tf.concat(x[key].values, axis=0)
    else:
      x = tf.concat(x.values, axis=0) 
  return x

def reduce_sum(x, axis=0):
  strategy = get_strategy()
  if strategy.num_replicas_in_sync > 1: 
    return strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=axis).numpy()
  return tf.reduce_sum(x, axis)

def reduce_mean(x, axis=0):
  strategy = get_strategy()
  if strategy.num_replicas_in_sync > 1: 
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=axis).numpy()
  return tf.reduce_mean(x, axis)

def sum_merge(x):
  strategy = get_strategy()
  if strategy.num_replicas_in_sync > 1:
    return tf.reduce_sum(tf.stack(x.values, axis=0), 0).numpy()
  return x.numpy()

def mean_merge(x):
  strategy = get_strategy()
  if strategy.num_replicas_in_sync > 1:
    return tf.reduce_mean(tf.stack(x.values, axis=0), 0).numpy()
  return x.numpy()

def tonumpy(*xs):
  strategy = get_strategy()
  res = []
  for x in xs:
    if strategy.num_replicas_in_sync > 1: 
      if isinstance(x, dict):
        for key in x:
          if hasattr(x[key], 'values'):
            x[key] = tf.concat(x[key].values, axis=0).numpy()
          else:
            # x[key] = tf.concat(x[key], axis=0).numpy()
            x[key] = x[key].numpy()
      else:
        if hasattr(x, 'values'):
          x = tf.concat(x.values, axis=0).numpy()
        else:
          # x = tf.concat(x, axis=0).numpy()
          x = x.numpy()
    else:
      if isinstance(x, dict):
        for key in x:
          x[key] = x[key].numpy()
      else:
        x = x.numpy()
    res += [x]
  if len(res) == 1:
    return res[0]
  else:
    return res
