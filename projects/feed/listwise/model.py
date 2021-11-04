#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2020-04-12 20:13:51.596792
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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input

import melt
import gezi 
logging = gezi.logging

class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__() 

    # self.encoder = melt.layers.transformer.Encoder(num_layers=5, d_model=16, num_heads=2, 
    #                                                dff=16, maximum_position_encoding=100, rate=1)

    self.encoder = tf.keras.layers.GRU(16, return_sequences=True, 
                                       dropout=0.1, recurrent_dropout=0.1)

    # self.pos_emb = keras.layers.Embedding(100, 16)
    self.dense = keras.layers.Dense(1)

  def call(self, input):
    click_feats = input['click_feats']
    bs = melt.get_shape(click_feats, 0)
    click_feats = tf.reshape(click_feats, [bs, -1, 16])
    # x_pos = self.pos_emb(input['positions'])
    # x = click_feats + x_pos
    x = click_feats 
    x = self.encoder(x)
    x = self.dense(x)
    x = tf.squeeze(x, -1)

    mask = tf.cast(input['positions'] > 0, tf.float32)

    x = x * mask

    return x
