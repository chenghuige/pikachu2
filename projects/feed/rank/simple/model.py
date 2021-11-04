#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-07-26 20:15:30.419843
#   \Description  TODO maybe input should be more flexible, signle feature, cross, cat, lianxu choumi
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
import melt

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

keras = tf.keras
from keras import backend as K

import numpy as np
from config import *

# output logits!
class Wide(keras.Model):
  def __init__(self):
    super(Wide, self).__init__()
    Embedding = melt.layers.Embedding
    self.emb = Embedding(FLAGS.feature_dict_size + 1, 1, name='emb')
    if FLAGS.wide_addval: 
      self.mult = keras.layers.Multiply()

  # put bias in build so we can trac it as WideDeep/Wide/bias
  def build(self, input_shape):
    self.bias = K.variable(value=[0.], name='bias')

  def call(self, input):
    indexes = input['index']
    values = input['value']

    x = tf.nn.embedding_lookup_sparse(params=self.emb(None), sp_ids=indexes, sp_weights=values, combiner='sum') 
    x = x + self.bias
    x = K.squeeze(x, -1)
    return x  

class Deep(keras.Model):
  def __init__(self):
    super(Deep, self).__init__()
    Embedding = melt.layers.Embedding
    self.emb = Embedding(FLAGS.feature_dict_size + 1, FLAGS.hidden_size, name='emb')
    self.emb_dim = FLAGS.hidden_size
    if FLAGS.field_emb:
      self.field_emb = Embedding(FLAGS.field_dict_size + 1, FLAGS.hidden_size, name='field_emb')
      self.emb_dim += FLAGS.hidden_size

    self.mult = keras.layers.Multiply()
    
    if FLAGS.use_doc_emb: 
      self.doc_dense = keras.layers.Dense(FLAGS.hidden_size, activation='relu')

    if not FLAGS.mlp_dims:
      self.mlp = None
    else:
      dims = [int(x) for x in FLAGS.mlp_dims.split(',')]
      activation = FLAGS.dense_activation 
      drop_rate = FLAGS.mlp_drop 
      self.mlp = melt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
    
    act = FLAGS.dense_activation if FLAGS.deep_final_act else None
    self.dense = keras.layers.Dense(1, activation=act)

  def call(self, input, training=False):
    indexes = input['index']
    values = input['value']
    fields = input['field']
    doc_emb = input['doc_emb']
    
    assert FLAGS.pooling == 'sum' or FLAGS.pooling == 'mean'
    assert not FLAGS.field_concat, "TODO.."
    values_ = values if FLAGS.deep_addval else None
    x = tf.nn.embedding_lookup_sparse(params=self.emb(None), sp_ids=indexes, sp_weights=values_, combiner=FLAGS.pooling)
    if FLAGS.field_emb:
      x = K.concatenate([x, tf.nn.embedding_lookup_sparse(params=self.field_emb(None), sp_ids=fields, sp_weights=None, combiner=FLAGS.pooling)], axis=-1)

    if FLAGS.use_doc_emb:
      x2 = self.doc_dense(doc_emb) 
      x = K.concatenate([x, x2], axis=1)

    if self.mlp:
      x = self.mlp(x, training=training)

    x = self.dense(x)
    x = K.squeeze(x, -1)
    return x

class WideDeep(keras.Model):   
  def __init__(self):
    super(WideDeep, self).__init__()
    self.wide = Wide()
    self.deep = Deep() 
    self.dense = keras.layers.Dense(1)

  def call(self, input, training=False):
    w = self.wide(input)
    d = self.deep(input, training=training)
    
    if FLAGS.deep_wide_combine == 'concat':
      x = K.stack([w, d], 1)
      x = self.dense(x)
      x = K.squeeze(x, -1)
    else:
      x = w + d
    
    return x
