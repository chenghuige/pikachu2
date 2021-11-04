#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   transformer.py
#        \author   chenghuige  
#          \date   2020-05-02
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from absl import flags
FLAGS = flags.FLAGS

import functools
import six
import re
from functools import partial
import traceback
import copy
import math
from functools import partial

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints

# import tensorflow_addons as tfa

import numpy as np

import gezi
import melt

logging = gezi.logging

from melt import dropout, softmax_mask
from melt.rnn import OutputMethod, encode_outputs

keras = tf.keras
layers = tf.keras.layers
Layer = layers.Layer

# https://www.tensorflow.org/tutorials/text/transformer

def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3.0))))

# TODO why... ValueError: Weights for model sequential have not yet been created. 
# Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.
# husky.main model.compile的时候
# def point_wise_feed_forward_network(d_model, dff):
#   return tf.keras.Sequential([
#       tf.keras.layers.Dense(dff, activation=gelu),  # (batch_size, seq_len, dff)
#       tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
#   ])

class point_wise_feed_forward_network(Layer):
  def __init__(self, d_model, dff, **kwargs):
    super(point_wise_feed_forward_network, self).__init__(**kwargs)
    self.dense1 = tf.keras.layers.Dense(dff, activation=gelu)
    self.dense2 = tf.keras.layers.Dense(d_model)
  
  def call(self, x):
    return self.dense2(self.dense1(x))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(Layer):
  def __init__(self, num_heads, d_model=None, return_att=True):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    if d_model:
      assert d_model % self.num_heads == 0
      self.depth = d_model // self.num_heads
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)

    self.return_att = return_att

  def build(self, input_shape):
    if not self.d_model:
      d_model = input_shape[-1]
      assert d_model % self.num_heads == 0, f'{d_model} {self.num_heads}'
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)
      self.depth = d_model // self.num_heads
      self.d_model = d_model
        
  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    seq_len = melt.get_shape(x, 1)
    x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, q, k, mask=None, v=None):
    if v is None:
      v = k
    if mask is not None:
      mask = tf.cast(mask, tf.float32)
      mask = mask[:, tf.newaxis, tf.newaxis, :]

    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    # print(q, k, v, mask)
    scaled_attention, attention_weights = melt.scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
      
    # seq_len = melt.get_shape(q, 1)
    # output = tf.reshape(output, [batch_size, seq_len, self.d_model])

    if self.return_att:
      return output, attention_weights
    else:
      return output

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, dff, rate=0.):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(num_heads, d_model)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.linear = tf.keras.layers.Dense(d_model)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, mask=None):
    attn_output, _ = self.mha(x, x, mask=mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(self.linear(x) + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)    
    
    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, num_heads, d_model, dff, 
               maximum_position_encoding=None, rate=0., **kwargs):
    super(Encoder, self).__init__(**kwargs)

    self.d_model = d_model
    self.num_layers = num_layers

    if maximum_position_encoding:
      self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
    else:
      self.pos_encoding = None
    
    self.enc_layers = [EncoderLayer(num_heads, d_model, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, sequence_length=None, pos_encoding=None, mask=None):
    batch_size = tf.shape(x)[0]
    seq_len = melt.get_shape(x, 1)

    if mask is None and sequence_length is not None:
      mask = (1. - tf.sequence_mask(sequence_length, maxlen=seq_len, dtype=tf.float32))
    
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    if pos_encoding is not None:
      x += pos_encoding
    elif self.pos_encoding is not None:
      x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, mask)

    x = tf.reshape(x, [batch_size, seq_len, self.d_model])
    
    return x  # (batch_size, input_seq_len, d_model)
    