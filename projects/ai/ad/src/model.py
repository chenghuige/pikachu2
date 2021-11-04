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

import numpy as np

import melt
import gezi 
logging = gezi.logging

# from config import *
from projects.ai.ad.src.config import *

# self.encoder = melt.layers.transformer.Encoder(num_layers=5, d_model=16, num_heads=2, 
#                                                dff=16, maximum_position_encoding=100, rate=1)
# self.encoder = tf.keras.layers.GRU(32, return_sequences=False, 
#                                    dropout=0.1, recurrent_dropout=0.1)

# class Baseline(keras.Model):
#   def __init__(self):
#     super(Baseline, self).__init__() 

#     self.cemb = tf.keras.layers.Embedding(5000000, FLAGS.emb_size, name='cemb')
#     self.encoder = melt.layers.Pooling('sum')
#     self.dense_age = keras.layers.Dense(1)
#     self.dense_gender = keras.layers.Dense(1)

#   def call(self, input):
#     # gezi.set('input', input)
#     creative_ids = input['creative_ids']
#     x = self.encoder(self.cemb(creative_ids))
#     self.age = self.dense_age(x)
#     self.gender = self.dense_gender(x)

#     self.pred_age = tf.math.sigmoid(self.age)
#     self.pred_gender = tf.math.sigmoid(self.gender) 

#     return self.gender

# class ClsTransformer(keras.Model):
#   def __init__(self):
#     super(ClsTransformer, self).__init__() 

#     # self.cemb = tf.keras.layers.Embedding(5000000, FLAGS.emb_size, name='cemb')
#     self.aemb = tf.keras.layers.Embedding(4000000, FLAGS.emb_size, name='aemb')
#     self.pemb = tf.keras.layers.Embedding(20, FLAGS.emb_size, name='pemb')
#     self.iemb = tf.keras.layers.Embedding(400, FLAGS.emb_size, name='iemb')
#     self.temb = tf.keras.layers.Embedding(100, FLAGS.emb_size, name='temb')
#     # self.ctemb = tf.keras.layers.Embedding(200, FLAGS.emb_size, name='ctemb')

#     self.encoder = melt.layers.transformer.Encoder(num_layers=FLAGS.num_layers, d_model=256, num_heads=FLAGS.num_heads, 
#                                                    dff=512, maximum_position_encoding=FLAGS.max_len + 1, rate=FLAGS.dropout)

#     self.combine = melt.layers.SemanticFusionCombine()
#     self.dense_age = keras.layers.Dense(10)
#     self.dense_gender = keras.layers.Dense(1)

#   def call(self, input):
#     # gezi.set('input', input)
#     LEN = FLAGS.max_len
#     x_in = input['ad_ids'][:,:LEN]
#     x_mask = tf.not_equal(x_in, 0)
#     x_len = melt.length(x_in)
#     # x_c = self.cemb(x_in)
#     x_a = self.aemb(x_in)
#     x_p = self.pemb(input['product_categories'][:,:LEN])
#     x_i = self.iemb(input['industries'][:,:LEN])
#     x_t = self.temb(input['times'][:,:LEN])
#     # x_ct = self.ctemb(input['click_times'][:,:5000])

#     x = tf.concat([x_a, x_p, x_i, x_t], axis=-1)
#     # x_other = tf.concat([x_p, x_i, x_t], axis=-1)
#     # x = self.combine(x_a, x_other)

#     x = self.encoder(x, mask=x_mask)
#     x = melt.layers.Pooling(FLAGS.pooling)(x, x_len)

#     self.age = self.dense_age(x)
#     self.gender = self.dense_gender(x)

#     # self.pred_age = tf.math.sigmoid(self.age)
#     self.pred_age = tf.argmax(self.age, axis=1)
#     self.pred_gender = tf.math.sigmoid(self.gender) 

#     return self.gender

# class ClsTransformer2(keras.Model):
#   def __init__(self):
#     super(ClsTransformer2, self).__init__() 

#     # self.cemb = tf.keras.layers.Embedding(5000000 + 2, FLAGS.emb_size, name='cemb')
#     self.aemb = tf.keras.layers.Embedding(4000000 + 2, FLAGS.emb_size, name='aemb')
#     self.pemb = tf.keras.layers.Embedding(20 + 2, FLAGS.emb_size, name='pemb')
#     self.iemb = tf.keras.layers.Embedding(400 + 2, FLAGS.emb_size, name='iemb')
#     self.temb = tf.keras.layers.Embedding(100 + 2, FLAGS.emb_size, name='temb')
#     # self.ctemb = tf.keras.layers.Embedding(200, FLAGS.emb_size, name='ctemb')

#     self.encoder = melt.layers.transformer.Encoder(num_layers=FLAGS.num_layers, d_model=256, num_heads=FLAGS.num_heads, 
#                                                    dff=512, maximum_position_encoding=FLAGS.max_len + 10, rate=FLAGS.dropout)

#     self.combine = melt.layers.SemanticFusionCombine()
#     self.dense_age = keras.layers.Dense(10)
#     self.dense_gender = keras.layers.Dense(1)

#   def call(self, input):
#     # gezi.set('input', input)
#     LEN = FLAGS.max_len
#     ad_ids = input['ad_ids'][:,:LEN]
#     dummy = tf.zeros_like(ad_ids)[:,:1]
#     mask = tf.cast(tf.not_equal(ad_ids, 0), tf.int64)
#     delta = mask * 2
#     ad_ids = tf.concat([dummy + 1, dummy + 2, ad_ids + delta], axis=-1)
#     product_categories = input['product_categories'][:,:LEN]
#     product_categories = tf.concat([dummy + 1, dummy + 2, product_categories + delta], axis=-1)
#     industries = input['industries'][:,:LEN]
#     industries = tf.concat([dummy + 1, dummy + 2, industries + delta], axis=-1)
#     times = input['times'][:,:LEN]
#     times = tf.concat([dummy + 1, dummy + 2, times + delta], axis=-1)

#     x_in = ad_ids
#     x_mask = tf.not_equal(x_in, 0)
#     x_len = melt.length(x_in)
#     # x_c = self.cemb(x_in)
#     x_a = self.aemb(x_in)
#     x_p = self.pemb(product_categories)
#     x_i = self.iemb(industries)
#     x_t = self.temb(times)
#     # x_ct = self.ctemb(input['click_times'][:,:5000])

#     x = tf.concat([x_a, x_p, x_i, x_t], axis=-1)
#     # x_other = tf.concat([x_p, x_i, x_t], axis=-1)
#     # x = self.combine(x_a, x_other)

#     x = self.encoder(x, mask=x_mask)
#     # x = melt.layers.Pooling(FLAGS.pooling)(x, x_len)
#     x_age = x[:, 0, :]
#     x_gender = x[:, 1, :]

#     self.age = self.dense_age(x_age)
#     self.gender = self.dense_gender(x_gender)

#     # self.pred_age = tf.math.sigmoid(self.age)
#     self.pred_age = tf.argmax(self.age, axis=1)
#     self.pred_gender = tf.math.sigmoid(self.gender) 

#     return self.gender

class ClsModel(keras.Model):
  def __init__(self):
    super(ClsModel, self).__init__() 

    # self.cemb = tf.keras.layers.Embedding(3420000, FLAGS.emb_size, name='cemb')

    if FLAGS.use_w2v:
      emb = np.load('../input/all/glove-min5/emb.npy')
      FLAGS.vocab_size = emb.shape[0]
      FLAGS.emb_size = emb.shape[1]
      self.aemb = tf.keras.layers.Embedding(FLAGS.vocab_size, FLAGS.emb_size, name='aemb',
                                            embeddings_initializer=tf.constant_initializer(emb),
                                            trainable=FLAGS.train_emb)
    else:
       self.aemb = tf.keras.layers.Embedding(FLAGS.vocab_size, FLAGS.emb_size, name='aemb',
                                             trainable=True)

    self.piemb = tf.keras.layers.Embedding(70000, FLAGS.emb_size, name='piemb')
    self.pemb = tf.keras.layers.Embedding(20, FLAGS.emb_size, name='pemb')
    self.iemb = tf.keras.layers.Embedding(400, FLAGS.emb_size, name='iemb')
    self.temb = tf.keras.layers.Embedding(100, FLAGS.emb_size, name='temb')
    # self.ctemb = tf.keras.layers.Embedding(200, FLAGS.emb_size, name='ctemb')

    Encoder = getattr(tf.keras.layers, FLAGS.encoder)
    Encoder = tf.compat.v1.keras.layers.CuDNNGRU
    # self.encoder = Encoder(FLAGS.hidden_size, return_sequences=True)
                          #  dropout=FLAGS.dropout, recurrent_dropout=FLAGS.rdropout)
    self.encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                        num_units=FLAGS.hidden_size, 
                                        keep_prob=1. - FLAGS.dropout,
                                        share_dropout=False,
                                        recurrent_dropout=True,
                                        concat_layers=FLAGS.concat_layers,
                                        bw_dropout=True,
                                        residual_connect=False,
                                        train_init_state=False,
                                        cell='lstm')

    # self.dropout = tf.keras.layers.Dropout(FLAGS.dropout)
    if FLAGS.lm_target:
      vsize = 1000000 if FLAGS.lm_target in ['ad_ids', 'creative_ids'] else 70000
      self.sampled_weight = self.add_weight(name='sampled_weight',
                                            shape=(vsize, FLAGS.hidden_size),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

      self.sampled_bias = self.add_weight(name='sampled_bias',
                                            shape=(vsize,),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

      self.softmax_loss_function = melt.seq2seq.gen_sampled_softmax_loss_function(100,
                                                                                  vsize,
                                                                                  weights=self.sampled_weight,
                                                                                  biases=self.sampled_bias,
                                                                                  log_uniform_sample=True,
                                                                                  is_predict=False,
                                                                                  sample_seed=1234)

    # self.combine = melt.layers.SemanticFusionCombine()
    self.dropout = keras.layers.Dropout(0.2)
    self.dense_age = keras.layers.Dense(10, name='dense_age')
    self.dense_gender = keras.layers.Dense(1, name='dense_gender')

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    gezi.set('input', input)
    LEN = FLAGS.max_len
    x_in = input['ad_ids'][:,:LEN]
    x_mask = tf.not_equal(x_in, 0)
    x_len = melt.length(x_in) if FLAGS.use_mask else None
    # x_c = self.cemb(input['creative_ids'][:,:LEN])
    x_a = self.aemb(x_in)
    # x_pi = self.piemb(input['product_ids'][:,:LEN])
    # x_p = self.pemb(input['product_categories'][:,:LEN])
    # x_i = self.iemb(input['industries'][:,:LEN])
    # x_t = self.temb(input['times'][:,:LEN])
    # x_ct = self.ctemb(input['click_times'][:,:5000])

    # x = tf.concat([x_a, x_p, x_pi, x_i, x_t], axis=-1)

    x = x_a

    # x_other = tf.concat([x_p, x_i, x_t], axis=-1)
    # x = self.combine(x_a, x_other)

    x = self.dropout(x)

    x = self.encoder(x, x_len)
    # x = self.encoder(x)

    # print(x)

    # x = self.dropout(x)

    if FLAGS.lm_target:
      return x

    x = self.pooling(x, x_len)

    self.age = self.dense_age(x)
    self.gender = self.dense_gender(x)

    # self.pred_age = tf.math.sigmoid(self.age)
    self.pred_age = tf.argmax(self.age, axis=1)
    self.pred_gender = tf.math.sigmoid(self.gender) 

    return self.age


class ClsModel2(keras.Model):
  def __init__(self):
    super(ClsModel2, self).__init__() 

    # self.cemb = tf.keras.layers.Embedding(3420000, FLAGS.emb_size, name='cemb')

    if FLAGS.use_w2v:
      emb = np.load('../input/all/glove-min5/emb.npy')
      FLAGS.vocab_size = emb.shape[0]
      FLAGS.emb_size = emb.shape[1]
      self.aemb = tf.keras.layers.Embedding(FLAGS.vocab_size, FLAGS.emb_size, name='aemb',
                                            embeddings_initializer=tf.constant_initializer(emb),
                                            trainable=FLAGS.train_emb)
    else:
       self.aemb = tf.keras.layers.Embedding(FLAGS.vocab_size, FLAGS.emb_size, name='aemb',
                                             trainable=FLAGS.train_emb)
    self.piemb = tf.keras.layers.Embedding(70000, FLAGS.emb_size, name='piemb')
    self.pemb = tf.keras.layers.Embedding(20, FLAGS.emb_size, name='pemb')
    self.iemb = tf.keras.layers.Embedding(400, FLAGS.emb_size, name='iemb')
    self.temb = tf.keras.layers.Embedding(100, FLAGS.emb_size, name='temb')
    # self.ctemb = tf.keras.layers.Embedding(200, FLAGS.emb_size, name='ctemb')

    self.position_emb = tf.keras.layers.Embedding(10000, 320, name='position_emb')

    # Encoder = getattr(tf.keras.layers, FLAGS.encoder)
    # self.encoder = Encoder(FLAGS.hidden_size, return_sequences=True, 
                          #  dropout=FLAGS.dropout, recurrent_dropout=FLAGS.rdropout)

    self.encoder = melt.layers.transformer.Encoder(num_layers=1, d_model=128, num_heads=4,
                                                   dff=128, rate=0)

    # self.encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
    #                                     num_units=FLAGS.hidden_size, 
    #                                     keep_prob=1. - FLAGS.dropout,
    #                                     share_dropout=False,
    #                                     recurrent_dropout=False,
    #                                     concat_layers=FLAGS.concat_layers,
    #                                     bw_dropout=False,
    #                                     residual_connect=False,
    #                                     train_init_state=False,
    #                                     cell='lstm')

    # self.dropout = tf.keras.layers.Dropout(FLAGS.dropout)
    if FLAGS.lm_target:
      vsize = 1000000 if FLAGS.lm_target in ['ad_ids', 'creative_ids'] else 70000
      self.sampled_weight = self.add_weight(name='sampled_weight',
                                            shape=(vsize, FLAGS.hidden_size),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

      self.sampled_bias = self.add_weight(name='sampled_bias',
                                            shape=(vsize,),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

      self.softmax_loss_function = melt.seq2seq.gen_sampled_softmax_loss_function(100,
                                                                                  vsize,
                                                                                  weights=self.sampled_weight,
                                                                                  biases=self.sampled_bias,
                                                                                  log_uniform_sample=True,
                                                                                  is_predict=False,
                                                                                  sample_seed=1234)

    self.mlp = melt.layers.MLP([128, 32], drop_rate=0.2, name='mlp')
    # self.combine = melt.layers.SemanticFusionCombine()
    self.dense_age = keras.layers.Dense(10, name='dense_age')
    self.dense_gender = keras.layers.Dense(1, name='dense_gender')

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    gezi.set('input', input)

    # tf.print(input['age'])
    # if K.learning_phase():
    #   with open('/tmp/1.txt', 'a') as out:
    #     print(gezi.decode(input['id'].numpy()).astype(int), file=out)

    # if K.learning_phase():
    #   print(gezi.decode(input['id'].numpy()).astype(int))

    LEN = FLAGS.max_len
    x_in = input['ad_ids'][:,:LEN]
    x_mask = tf.not_equal(x_in, 0)
    x_len = melt.length(x_in) if FLAGS.use_mask else None
    # x_c = self.cemb(input['creative_ids'][:,:LEN])
    x_a = self.aemb(x_in)
    # x_pi = self.piemb(input['product_ids'][:,:LEN])
    # x_p = self.pemb(input['product_categories'][:,:LEN])
    # x_i = self.iemb(input['industries'][:,:LEN])
    # x_t = self.temb(input['times'][:,:LEN])
    # # x_ct = self.ctemb(input['click_times'][:,:5000])

    # x = tf.concat([x_a, x_p, x_pi, x_i, x_t], axis=-1)

    x = x_a

    # x_other = tf.concat([x_p, x_i, x_t], axis=-1)
    # x = self.combine(x_a, x_other)

    x += self.position_emb(melt.get_positions(x))
    x = self.encoder(x)

    # print(x)

    # x = self.dropout(x)

    if FLAGS.lm_target:
      return x

    x = self.pooling(x, x_len)

    # x = self.mlp(x)

    self.age = self.dense_age(x)
    self.gender = self.dense_gender(x)

    # self.pred_age = tf.math.sigmoid(self.age)
    self.pred_age = tf.argmax(self.age, axis=1)
    self.pred_gender = tf.math.sigmoid(self.gender) 

    return self.age


