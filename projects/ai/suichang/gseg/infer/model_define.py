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

def resize(image, image_size, pad=False):
  return tf.image.resize(image, image_size)

def init_model():
  model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
  print('model_path', model_path)
  custom_objects={'tf': tf, 
  'swish_activation': swish_activation, 
  'FixedDropout': FixedDropout,
  'relu6': tf.nn.relu6,
  'resize': resize,}
  model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
  input = model.input
  # input._name = 'image'
  out = tf.argmax(model.output, -1)
  mask = tf.cast(out < 4, tf.int64)
  out = (out + 1) * mask + (out + 3) * (1 - mask)
  out = tf.expand_dims(out, -1)
  # out = tf.identity(out, name='pred')
  model = tf.keras.models.Model(input, out)
  model.summary()
  return model
  
