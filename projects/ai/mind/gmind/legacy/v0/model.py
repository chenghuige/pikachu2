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
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization

import numpy as np

import melt
import gezi 
logging = gezi.logging

from projects.ai.mind.src.config import *
from projects.ai.mind.src import util
from projects.ai.mind.src import loss

class TitleEncoder(keras.Model):
  def __init__(self, word_emb):
    super(TitleEncoder, self).__init__()   
    self.word_emb = word_emb
    self.pooling = melt.layers.Pooling(FLAGS.title_pooling)
    self.output_dim = word_emb.output_dim

  # in eger mode ok without below but graph not
  def compute_output_shape(self, input_shape):
    return (None, self.output_dim)

  def call(self, title):
    return self.pooling(self.word_emb(title))

class TitlesEncoder(keras.Model):
  def __init__(self, title_encoder):
    super(TitlesEncoder, self).__init__()   
    self.encoder =  keras.layers.TimeDistributed(title_encoder)
    self.pooling = util.get_att_pooling('din')

  def call(self, titles, tlen, query=None):
    titles = self.encoder(titles)
    # print(query, titles, tlen)
    return self.pooling(query, titles, tlen)

class DocEncoder(keras.Model):
  pass

class HistoryEncoder(keras.Model):
  pass

class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__() 
    self.mode = 'train'
    
    self.input_ = {}

    def _emb(vocab_name, emb_name=None):
      return util.create_emb(vocab_name, emb_name)

    self.uemb = _emb('uid')
    self.demb = _emb('did')

    self.cat_emb = _emb('cat')
    self.scat_emb = _emb('sub_cat')
    self.entity_emb = _emb('entity')
    self.entity_type_emb = _emb('entity_type')
    self.word_emb = _emb('word')

    self.hour_emb = Embedding(24, FLAGS.emb_size, name='hour_emb')
    self.weekday_emb = Embedding(7, FLAGS.emb_size, name='weekday_emb')
    self.fresh_hour_emb = Embedding(300, FLAGS.emb_size, name='fresh_hour_emb') # 7 * 24
    self.fresh_day_emb = Embedding(50, FLAGS.emb_size, name='fresh_day_emb')
    self.position_emb = Embedding(300, FLAGS.emb_size, name='position_emb')

    self.title_lookup = melt.layers.LookupArray(FLAGS.title_lookup)
    self.doc_lookup = melt.layers.LookupArray(FLAGS.doc_lookup)

    self.title_encoder = TitleEncoder(self.word_emb)
    self.titles_encoder = TitlesEncoder(self.title_encoder)

    self.sum_pooling = melt.layers.SumPooling()
    self.mean_pooling = melt.layers.MeanPooling()
    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.feat_pooling = melt.layers.Pooling(FLAGS.feat_pooling)
    self.his_simple_pooling = melt.layers.Pooling(FLAGS.his_simple_pooling)

    self.dense = Dense(1) if not FLAGS.use_multi_dropout else melt.layers.MultiDorpout(1, drop_rate=0.3)
    self.batch_norm = BatchNormalization()
    self.dropout = keras.layers.Dropout(FLAGS.dropout)
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    mlp_dims = [FLAGS.emb_size * 2, FLAGS.emb_size] if not FLAGS.big_mlp else [FLAGS.emb_size * 4, FLAGS.emb_size * 2, FLAGS.emb_size]
    self.dense_mlp = melt.layers.MLP(mlp_dims,
                                     activation=activation, 
                                     drop_rate=FLAGS.mlp_dropout,
                                     name='dense_mlp')
    
    mlp_dims = [512, 256, 64] if not FLAGS.big_mlp else [1024, 512, 256]
    self.mlp = melt.layers.MLP(mlp_dims, activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_encoder = util.get_encoder(FLAGS.his_encoder)
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)
    self.his_pooling = util.get_att_pooling(FLAGS.his_pooling)
    self.his_pooling2 = util.get_att_pooling(FLAGS.his_pooling2)
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

    if FLAGS.his_strategy.startswith('bst'):
      self.transformer = melt.layers.transformer.Encoder(num_layers=1, d_model=FLAGS.hidden_size, num_heads=FLAGS.num_heads, 
                                                         dff=FLAGS.hidden_size, maximum_position_encoding=None, activation=FLAGS.transformer_activation,
                                                         rate=FLAGS.transformer_dropout)

    self.fusion = melt.layers.SemanticFusion(drop_rate=0.1)

    if FLAGS.feat_pooling == 'cin':
      from deepctr.layers.interaction import CIN
      self.cin = CIN((128, 128,), 'relu', True, 0, 1024)
      self.feat_pooling = self.cin

    if FLAGS.aux_loss_rate or FLAGS.lm_target:
      vsize = gezi.get('vocab_sizes')['vid'][0]
      # hidden_size = FLAGS.hidden_size if FLAGS.his_encoder in ['lstm', 'gru'] else  int(FLAGS.hidden_size / 2)
      hidden_size = int(FLAGS.hidden_size / 2)
      self.sampled_weight = self.add_weight(name='sampled_weight',
                                            shape=(vsize, hidden_size),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

      self.sampled_bias = self.add_weight(name='sampled_bias',
                                            shape=(vsize,),
                                            #initializer = keras.initializers.RandomUniform(minval=-10, maxval=10, seed=None),
                                            dtype=tf.float32,
                                            trainable=True)

      self.softmax_loss_function = melt.seq2seq.gen_sampled_softmax_loss_function(5,
                                                                                  vsize,
                                                                                  weights=self.sampled_weight,
                                                                                  biases=self.sampled_bias,
                                                                                  log_uniform_sample=True,
                                                                                  is_predict=False,
                                                                                  sample_seed=1234)

  def deal_dense(self, input):
    feats = []
    title_len_ = melt.scalar_feature(tf.cast(input['title_len'], tf.float32), max_val=20, scale=True)
    abstract_len_ = melt.scalar_feature(tf.cast(input['abstract_len'], tf.float32), max_val=120, scale=True)
   
    his_len_ = melt.scalar_feature(input['hist_len'], max_val=800, scale=True)
    impression_len_ = melt.scalar_feature(input['impression_len'], max_val=300, scale=True)
    feats += [
      title_len_,
      abstract_len_,
      his_len_,
      impression_len_
    ]

    if FLAGS.use_fresh:
      fresh = tf.cast(input['fresh'], tf.float32)
      fresh_hour_ = melt.scalar_feature(fresh / (3600), max_val=7 * 24, scale=True)
      fresh_day_ = melt.scalar_feature(fresh / (3600 * 12), max_val=7 * 2, scale=True)
      feats += [
        fresh_hour_,
        fresh_day_,
      ]

    if FLAGS.use_position:
      position_rel_ = melt.scalar_feature((input['position'] + 1) / (input['impression_len'] + 1))
      position_ = melt.scalar_feature((input['position'] + 1) / (300 + 1))
      feats += [
        position_rel_,
        position_
        ]

    feats = tf.concat(feats, -1)
    dense_emb = self.dense_mlp(feats)
    return dense_emb

  # def build(self, input_shape=None):
  #  if input_shape is None:
  #     raise ValueError('You must provide an `input_shape` argument.')
  #  input_shape = tuple(input_shape)
  #  self._build_input_shape = input_shape
  #  super(Model, self).build(input_shape)
 
  #  self.built = True

  # @tf.function
  def call(self, input):
    # TODO tf2 keras seem to auto append last dim so need this
    melt.try_squeeze_dim(input)

    if not FLAGS.batch_parse:
      util.adjust(input, self.mode)

    # print(input)
  
    embs = []

    if 'history' in input:
      hlen = melt.length(input['history'])
      hlen = tf.math.maximum(hlen, 1)
      
    bs = melt.get_shape(input['did'], 0)

    # user 
    if FLAGS.use_uid:
      uemb = self.uemb(input['uid'])
      embs += [uemb]

    if FLAGS.use_did:
      demb = self.demb(input['did'])
      embs += [demb]

    if FLAGS.use_time_emb:
      embs += [
        self.hour_emb(input['hour']), 
        self.weekday_emb(input['weekday']),
      ]

    if FLAGS.use_fresh_emb:
      fresh = input['fresh']
      fresh_day = tf.cast(fresh / (3600 * 12), fresh.dtype)
      fresh_hour = tf.cast(fresh / 3600, fresh.dtype)
      embs += [
        self.fresh_day_emb(fresh_day),
        self.fresh_hour_emb(fresh_hour)
      ]

    if FLAGS.use_position_emb:
      embs += [
        self.position_emb(input['position'])
      ]

    if FLAGS.use_news_info and 'cat' in input:
      # print('------entity_emb', self.entity_emb.emb.weights) # check if trainable is fixed in eager mode
      embs += [
        self.cat_emb(input['cat']),
        self.scat_emb(input['sub_cat']),
        self.pooling(self.entity_type_emb(input['title_entity_types']), melt.length(input['title_entity_types'])),
        self.pooling(self.entity_type_emb(input['abstract_entity_types']), melt.length(input['abstract_entity_types'])),
      ]
      if FLAGS.use_entities and 'title_entities' in input:
        embs += [
          self.pooling(self.entity_emb(input['title_entities']), melt.length(input['title_entities'])),
          self.pooling(self.entity_emb(input['abstract_entities']), melt.length(input['abstract_entities'])),
        ]

    if FLAGS.use_history_info and 'history_cats' in input:
      embs += [
        self.his_simple_pooling(self.cat_emb(input['history_cats']), melt.length(input['history_cats'])),
        self.his_simple_pooling(self.scat_emb(input['history_sub_cats']), melt.length(input['history_sub_cats'])),
      ]
      if FLAGS.use_history_entities:
        try:
          embs += [
            self.his_simple_pooling(self.entity_type_emb(input['history_title_entity_types']), melt.length(input['history_title_entity_types'])),
            self.his_simple_pooling(self.entity_type_emb(input['history_abstract_entity_types']), melt.length(input['history_abstract_entity_types'])),
          ]    
          if FLAGS.use_entities and 'title_entities' in inpout:
            embs += [
              self.his_simple_pooling(self.entity_emb(input['history_title_entities']), melt.length(input['history_title_entities'])),
              self.his_simple_pooling(self.entity_emb(input['history_abstract_entities']), melt.length(input['history_abstract_entities'])),
            ]
        except Exception:
          pass  

    if FLAGS.use_history and FLAGS.use_did:
      dids = input['history']

      if FLAGS.his_strategy == 'bst' or FLAGS.his_pooling == 'mhead':
        mask = tf.cast(tf.equal(dids, 0), dids.dtype)
        dids += mask
        hlen = tf.ones_like(hlen) * 50
      hembs = self.demb(dids)

      his_embs = hembs
      his_embs = self.his_encoder(his_embs, hlen)
      self.his_embs = his_embs
    
      his_emb = self.his_pooling(demb, his_embs, hlen)
      
      embs += [his_emb]

    if FLAGS.use_title:
      cur_title = self.title_encoder(self.title_lookup(input['ori_did']))
      dids = input['ori_history']
      if FLAGS.max_titles:
        dids = dids[:, :FLAGS.max_titles]
      his_title = self.titles_encoder(self.title_lookup(dids), hlen, cur_title)
      embs += [
        cur_title,
        his_title
      ]

    # 用impression id 会dev test不一致 不直接用id
    if FLAGS.use_impressions:
      embs += [
        self.mean_pooling(self.demb(input['impressions']))
      ]

    if FLAGS.use_dense:
      dense_emb = self.deal_dense(input)
      embs += [dense_emb]
    
    # logging.debug('-----------embs:', len(embs))
    embs = tf.stack(embs, axis=1)

    if FLAGS.batch_norm:
      embs = self.batch_norm(embs)

    if FLAGS.l2_normalize_before_pooling:
      x = tf.math.l2_normalize(embs)

    x = self.feat_pooling(embs)

    if FLAGS.dropout:
      x = self.dropout(x)

    if FLAGS.use_dense:
      x = tf.concat([x, dense_emb], axis=1)

    if FLAGS.use_his_concat:
      x = tf.concat([x, his_concat], axis=1)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.impression_id = input['impression_id']
    self.position = input['position']
    self.history_len = input['hist_len']
    self.impression_len = input['impression_len']
    self.input_ = input
    return self.logit

  # @tf.function
  def get_loss(self):
    loss_fn_ = getattr(loss, FLAGS.loss)
    def loss_fn(y_true, y_pred):
      return loss_fn_(y_true, y_pred, self.input_, self)

    return loss_fn

  def get_model(self, inputs):
    inputs_ = inputs.copy()
    # x = {}
    # keys = ['did']
    # for key in keys:
    #   x[key] = inputs_[key]
    out = self.call(inputs)
    # out = self.call(x)
    model = keras.Model(inputs_, outputs=out)
    # model = keras.Model(inputs=list(inputs_.values()), outputs=out)
    return model
