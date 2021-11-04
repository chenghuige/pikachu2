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

from collections import OrderedDict

import tensorflow as tf
import tensorflow_hub as hub
from transformers import TFAutoModel

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input

import melt
import gezi 
logging = gezi.logging

from config import *
from utils import *

def bert_model(trainable_bert=True):
    """Build and return a multilingual BERT model and tokenizer."""
    max_seq_length = FLAGS.max_len
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="all_segment_id")
    
    # Load a SavedModel on TPU from GCS. This model is available online at 
    # https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1. You can use your own 
    # pretrained models, but will need to add them as a Kaggle dataset.
    bert_layer = tf.saved_model.load(FLAGS.pretrained)
    # Cast the loaded model to a TFHub KerasLayer.
    bert_layer = hub.KerasLayer(bert_layer, trainable=trainable_bert)
 
    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
    output = tf.keras.layers.Dense(32, activation='relu')(pooled_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels')(output)

    # TODO 为什么这里能写成dict ？ 下面 xlm_model 就不行呢？ 因为是混合type？
    return tf.keras.Model(inputs={'input_word_ids': input_word_ids,
                                  'input_mask': input_mask,
                                  'all_segment_id': segment_ids},
                          outputs=output)

class BertModel(keras.Model):
  def __init__(self):
    super(BertModel, self).__init__() 

    bert_layer = hub.load(FLAGS.pretrained)
    bert_layer = hub.KerasLayer(bert_layer, trainable=True)
    self.bert_layer = bert_layer
    dims = [32]
    self.mlp = melt.layers.MLP(dims)

    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
    self.dense = keras.layers.Dense(odim, activation='sigmoid')

  def call(self, input):
    input_word_ids = input['input_word_ids']
    input_mask = input['input_mask']
    segment_ids = input['all_segment_id']
    x, _ = self.bert_layer([input_word_ids, input_mask, segment_ids])
    x = self.mlp(x)
    x = self.dense(x)
    return x

class BowModel(keras.Model):
  def __init__(self):
    super(BowModel, self).__init__() 
    NUM_WORDS = 260000 # 250002
    self.emb = keras.layers.Embedding(NUM_WORDS, 128)
    self.pooling = melt.layers.Pooling('max')
    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
    self.dense = keras.layers.Dense(odim, activation=None)

  def build(self, input_shape):
    self.built = True
    
  def call(self, input):
    input_word_ids = input['input_word_ids']
    x = self.emb(input_word_ids)
    x = self.pooling(x)
    x = self.dense(x)
    return x

class BowAttModel(keras.Model):
  def __init__(self):
    super(BowAttModel, self).__init__() 
    NUM_WORDS = 260000 # 250002
    self.emb = keras.layers.Embedding(NUM_WORDS, 128)
    self.pooling = melt.layers.Pooling('att')
    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
    self.dense = keras.layers.Dense(odim, activation=None)

  def call(self, input):
    input_word_ids = input['input_word_ids']
    x = self.emb(input_word_ids)
    x = self.pooling(x)
    x = self.dense(x)
    return x

class LstmModel(keras.Model):
  def __init__(self):
    super(LstmModel, self).__init__() 
    NUM_WORDS = 260000 # 250002
    hidden_size = 128
    self.emb = keras.layers.Embedding(NUM_WORDS, hidden_size)
    self.encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1)
    self.pooling = melt.layers.Pooling('max')
    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
    self.dense = keras.layers.Dense(odim, activation=None)
    
  def call(self, input):
    input_word_ids = input['input_word_ids']
    x = self.emb(input_word_ids)
    x = self.encoder(x)
    x = self.pooling(x)
    x = self.dense(x)
    return x

class LstmAttModel(keras.Model):
  def __init__(self):
    super(LstmAttModel, self).__init__() 
    NUM_WORDS = 260000 # 250002
    hidden_size = 128
    self.emb = keras.layers.Embedding(NUM_WORDS, hidden_size)
    self.encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=0.1)
    self.pooling = melt.layers.Pooling('att')
    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
    self.dense = keras.layers.Dense(odim, activation=None)
    
  def call(self, input):
    input_word_ids = input['input_word_ids']
    x = self.emb(input_word_ids)
    x = self.encoder(x)
    x = self.pooling(x)
    x = self.dense(x)
    return x

class XlmModel(keras.Model):
  def __init__(self):
    super(XlmModel, self).__init__() 

    pretrained = FLAGS.pretrained

    with gezi.Timer(f'load xlm_model from {pretrained}', True, logging.info):
      self.transformer = TFAutoModel.from_pretrained(pretrained)
    self.transformer.trainable = False if FLAGS.freeze_pretrained else True
    
    self.base_pooling = melt.layers.Pooling(FLAGS.base_pooling)
    if FLAGS.use_mlp:
      self.mlp = melt.layers.MLP([256,128])

    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1

    if FLAGS.pooling != 'concat':
      self.pooling = melt.layers.Pooling(FLAGS.pooling)

    if not FLAGS.use_multi_dropout:
      self.dense = keras.layers.Dense(odim, activation='sigmoid')
    else:
      self.num_experts = 5
      self.denses1 = [keras.layers.Dense(32, activation='relu')] * self.num_experts
      self.denses2 = [keras.layers.Dense(odim, activation='sigmoid')] * self.num_experts
      self.dropouts = [keras.layers.Dropout(FLAGS.dropout)] * self.num_experts

  def call(self, input):
    # tf.print(input)
    input_word_ids = input['input_word_ids']
    x = self.transformer(input_word_ids)[0]
    x = self.base_pooling(x)

    # mmoe ?
    if FLAGS.use_word_ids2:
      x1 = x
      input_word_ids2 = input['input_word_ids2']
      x2 = self.transformer(input_word_ids2)[0]
      x2 = x2[:, 0, :]
      if FLAGS.pooling == 'concat':
        x = tf.concat([x, x2], -1)
      else: 
        # go here
        x = tf.stack([x, x2], axis=1)
        x = self.pooling(x)

    if FLAGS.use_mlp:
      x = self.mlp(x)

    if not FLAGS.use_multi_dropout:
      x = self.dense(x)
    else:
      xs = []
      for i in range(self.num_experts):
        x_i = self.dropouts[i](x)
        x_i = self.denses1[i](x_i)
        x_i = self.denses2[i](x_i)
        xs += [x_i]
        
      x = tf.reduce_mean(tf.concat(xs, axis=1), 1, keepdims=True)

    # tf.print(x)
    return x

class XlmTranslateModel(keras.Model):
  def __init__(self):
    super(XlmTranslateModel, self).__init__() 

    pretrained = FLAGS.pretrained

    with gezi.Timer(f'load xlm_model from {pretrained}', True, logging.info):
      self.transformer = TFAutoModel.from_pretrained(pretrained)

  def call(self, input):
    input_word_ids = input['input_word_ids']
    x = self.transformer(input_word_ids)[0]
    x = x[:, 0, :]
    x = tf.nn.l2_normalize(x, 1)

    input_word_ids2 = input['input_word_ids2']
    x2 = self.transformer(input_word_ids2)[0]
    x2 = tf.nn.l2_normalize(x2, 1)
    x2 = x2[:, 0, :]    

    x2_neg = tf.concat([x2[1:,:], x2[0:1,:]], axis=0)

    pos_score = tf.matmul(x, x2, transpose_b=True)
    neg_score = tf.matmul(x, x2_neg, transpose_b=True)

    return {'pos': pos_score, 'neg': neg_score}

## Old model just for compat 
def xlm_old_model():
  pretrained = FLAGS.pretrained 
  with gezi.Timer(f'load xlm_model from {pretrained}', True, print):
    transformer = TFAutoModel.from_pretrained(pretrained)
  input_word_ids = tf.keras.Input(shape=(FLAGS.max_len,), dtype=tf.int32, name="input_word_ids")
  sequence_output = transformer(input_word_ids)[0]
  cls_token = sequence_output[:, 0, :]
  odim = len(toxic_types) + 1 if FLAGS.multi_head else 1
  out = tf.keras.layers.Dense(odim, activation='sigmoid')(cls_token)

  model = tf.keras.Model(inputs=input_word_ids, outputs=out)

  return model

def xlm_model():
  inputs = {
            'input_word_ids': Input(shape=(FLAGS.max_len,), dtype=tf.int32, name="input_word_ids"),
            }
  if FLAGS.use_word_ids2:
    inputs['input_word_ids2'] = Input(shape=(FLAGS.max_len,), dtype=tf.int32, name="input_word_ids2")
  out = XlmModel()(inputs)

  # TODO why can not just inputs..
  model = keras.Model(inputs=list(inputs.values()), outputs=out)

  return model

def xlm_translate_model():
  inputs = {
            'input_word_ids': Input(shape=(FLAGS.max_len,), dtype=tf.int32, name="input_word_ids"),
            }
  if FLAGS.use_word_ids2:
    inputs['input_word_ids2'] = Input(shape=(FLAGS.max_len,), dtype=tf.int32, name="input_word_ids2"),
  out = XlmTranslateModel()(inputs)

  # TODO why can not just inputs..
  model = keras.Model(inputs=list(inputs.values()), outputs=out)

  return model


class XlmDetectLangModel(keras.Model):
  def __init__(self):
    super(XlmDetectLangModel, self).__init__() 

    pretrained = FLAGS.pretrained 
    with gezi.Timer(f'load xlm_model from {pretrained}', True, logging.info):
      self.transformer = TFAutoModel.from_pretrained(pretrained)

    if FLAGS.freeze_pretrained:
      self.transformer.trainable = False
    # dims = [32]
    # self.mlp = melt.layers.MLP(dims)

    odim = len(langs)
    self.dense = keras.layers.Dense(odim, activation='softmax')

  def call(self, input):
    input_word_ids = input['input_word_ids']
    x = self.transformer(input_word_ids)[0]
    x = x[:, 0, :]
    x = self.dense(x)
    return x

