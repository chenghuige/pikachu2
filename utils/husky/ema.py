#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ema.py
#        \author   chenghuige
#          \date   2021-01-25 17:11:24.238355
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

from tensorflow.keras import backend as K

# https://spaces.ac.cn/archives/6575

class ExponentialMovingAverage:
  """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
  def __init__(self, model, momentum=0.9999):
    self.momentum = momentum
    self.model = model
    self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

  def inject(self):
    """添加更新算子到model.metrics_updates。
        """
    self.initialize()
    for w1, w2 in zip(self.ema_weights, self.model.weights):
      op = K.moving_average_update(w1, w2, self.momentum)
      self.model.metrics_updates.append(op)

  def initialize(self):
    """ema_weights初始化跟原模型初始化一致。
        """
    self.old_weights = K.batch_get_value(self.model.weights)
    K.batch_set_value(zip(self.ema_weights, self.old_weights))

  def apply_ema_weights(self):
    """备份原模型权重，然后将平均权重应用到模型上去。
        """
    self.old_weights = K.batch_get_value(self.model.weights)
    ema_weights = K.batch_get_value(self.ema_weights)
    K.batch_set_value(zip(self.model.weights, ema_weights))

  def reset_old_weights(self):
    """恢复模型到旧权重。
        """
    K.batch_set_value(zip(self.model.weights, self.old_weights))
