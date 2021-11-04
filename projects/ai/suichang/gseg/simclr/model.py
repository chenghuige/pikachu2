#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2020-11-14 14:51:01.621399
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import flags
FLAGS = flags.FLAGS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
import melt as mt
from ..augment import augment
from .losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
from . import helpers

class Model(mt.Model):
  def __init__(self, temperature=0.1, backbone='EfficientNetB4', weights='noisy-student', 
               image_size=[256, 256], **kwargs):
    super(Model, self).__init__(**kwargs)
    self.temperature = temperature #
    Model_ = mt.image.get_classifier(backbone)
    self.model = Model_(include_top=False, weights=weights, input_shape=(*image_size, 3))
    self. preprocess = mt.image.get_preprocessing(backbone)
    self.pooling = keras.layers.GlobalAveragePooling2D()
    self.mlp = mt.layers.MLP([256, 128, 50], activate_last=False)
    self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

  def forward(self, x):
    return self.mlp(self.pooling(self.model(x, training=True)))
    
  def call(self, input):
    temperature = self.temperature
    image = input['image']
    a = augment(image)[0]
    b = augment(image)[0]
    a = self.preprocess(a)
    b = self.preprocess(b)

    bs = mt.get_shape(image, 0)
    negative_mask = helpers.get_negative_mask(bs)

    zis = self.mlp(self.pooling(self.model(a, training=True)))
    zjs = self.mlp(self.pooling(self.model(b, training=True)))

    # normalize projection feature vectors
    zis = tf.math.l2_normalize(zis, axis=1)
    zjs = tf.math.l2_normalize(zjs, axis=1)

    # (8, 50)
    # print(zis.shape, zjs.shape)

    l_pos = sim_func_dim1(zis, zjs)
    l_pos = tf.reshape(l_pos, (bs, 1))
    l_pos /= temperature

    # (8, 1)
    # print(l_pos.shape)

    negatives = tf.concat([zjs, zis], axis=0)

    # (16, 50)
    # print(negatives.shape)

    loss = 0

    for positives in [zis, zjs]:
      l_neg = sim_func_dim2(positives, negatives)
      labels = tf.zeros(bs, dtype=tf.int32)
      l_neg = tf.boolean_mask(l_neg, negative_mask)
      l_neg = tf.reshape(l_neg, (bs, -1))
      l_neg /= temperature
      # (8, 14)
      #   print(l_neg.shape)
      logits = tf.concat([l_pos, l_neg], axis=1) 
      loss += self.criterion(y_pred=logits, y_true=labels)
      loss = loss / (2 * bs)

    return loss

  def get_loss(self):
    return lambda y_true, y_pred: y_pred

  def get_model(self):
    if not FLAGS.dynamic_image_size:
      img_input = Input(shape=(*FLAGS.ori_image_size, 3), name='image')    
    else:
      img_input = Input(shape=(None, None, 3), name='image')

    inp = {'image': img_input}
    out = self.forward(inp)
    model = keras.Model(img_input, out, name=f'SimCLR_{FLAGS.backbone}')
    model.summary()
    return model

        
