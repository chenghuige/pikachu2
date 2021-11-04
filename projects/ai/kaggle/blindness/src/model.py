#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-09-02 21:30:07.433860
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

# import sys 
# import os

#keras = tf.keras
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,
                          Multiply, Lambda)
from tensorflow.keras import backend as K

from config import *

import gezi
logging = gezi.logging

def get_model(name): 
  if 'densenet' in name.lower():
    return getattr(tf.keras.applications.densenet, name)
  else:
    raise ValueError(name)

def final_output_fn(x):
  if 'linear_regression' in loss_type:
    final_output = Dense(1, activation='linear', name='final_output')(x)
  elif 'sigmoid_regression' in loss_type:
    x = Dense(1, activation='sigmoid')(x)
    #final_output = Multiply(name='final_output')([x, tf.constant(10.)])
    #https://github.com/keras-team/keras/issues/10204
    #final_output = Lambda(lambda x: x * 10.0, name='final_output')(x)
    final_output = Lambda(lambda x: x * 4.0, name='final_output')(x)
  elif 'sigmoid2_regression' in loss_type:
    final_output = Dense(1, activation='sigmoid', name='final_output')(x)
  elif loss_type == 'ordinal_classification':
    final_output = Dense(NUM_CLASSES, name='final_output')(x)
  elif loss_type == 'ordinal2_classification':
    final_output = Dense(NUM_CLASSES - 1, name='final_output')(x)
  else:
    final_output = Dense(NUM_CLASSES, name='final_output')(x)
  return final_output

class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    BaseModel = get_model(FLAGS.model)
    base_model = BaseModel(include_top=False,
                           weights=None,
                           input_shape=IMG_SHAPE)
    gezi.sprint(base_model)
    # base_model.load_weights(FLAGS.pretrain_path)
    # base_model.train_able = False
    self.base_model = base_model

    self.avg_pooling = GlobalAveragePooling2D() 
    self.drop_out = Dropout(0.5)
    self.dense = Dense(1024, activation='relu')

    self.out_dense = Dense(NUM_CLASSES, name='final_output')

    self.restore()

  # TODO how can pass path like init(path) ? in melt.flow ?
  # NOTICE restore is called after sess.run(init_op)
  def restore(self):
    logging.info(f'load weights from {FLAGS.pretrain_path}')
    self.base_model.load_weights(FLAGS.pretrain_path)

  def call(self, input):
    image = input['image']
    x = self.base_model(image)
    x = self.avg_pooling(x)
    x = self.drop_out(x)
    x = self.dense(x)
    x = self.drop_out(x)
    # x = final_output_fn(x)
    x = self.out_dense(x)

    return x
