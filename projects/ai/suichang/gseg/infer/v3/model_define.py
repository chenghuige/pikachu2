#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model_define.py
#        \author   chenghuige  
#          \date   2020-10-26 23:56:10.601464
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation

def swish_activation(x):
  return (K.sigmoid(x) * x)

class FixedDropout(tf.keras.layers.Dropout):
  def _get_noise_shape(self, inputs):
    if self.noise_shape is None:
      return self.noise_shape

    symbolic_shape = K.shape(inputs)
    noise_shape = [symbolic_shape[axis] if shape is None else shape
                    for axis, shape in enumerate(self.noise_shape)]
    return tuple(noise_shape)

def init_model():
  model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
  print('model_path', model_path)
  model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf, 'swish_activation': swish_activation, 'FixedDropout': FixedDropout}, compile=False)
  model.summary()
  return model
  
