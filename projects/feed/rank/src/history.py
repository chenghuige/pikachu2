#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   history.py
#        \author   chenghuige  
#          \date   2020-06-18 18:19:25.040631
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
import melt as mt

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

from tensorflow import keras
from tensorflow.keras import backend as K

import numpy as np
from projects.feed.rank.src.config import *
from projects.feed.rank.src import util 
import gezi 
logging = gezi.logging

class History(mt.Model):
  def __init__(self, doc_emb, topic_emb, kw_emb):
    super(History, self).__init__()
  
    self.doc_emb = doc_emb
    self.topic_emb = topic_emb
    self.kw_emb = kw_emb

    # self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)   
    self.regularizer = None
    Embedding = keras.layers.Embedding
    SimpleEmbedding = mt.layers.SimpleEmbedding
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()
    kwargs = dict(num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner, 
                  embeddings_regularizer=self.regularizer, num_shards=FLAGS.num_shards)
    self.kwargs = kwargs
    self.HashEmbedding = HashEmbedding

    if FLAGS.history_attention:
      self.history_att = mt.layers.DINAttention(attention_hidden_units=FLAGS.attention_mlp_dims, attention_activation=FLAGS.attention_activation, 
                                                  mode=FLAGS.attention_mode, weight_normalization=FLAGS.attention_norm, name="din_attention")
      self.rec_emb = SimpleEmbedding(2000, FLAGS.other_emb_dim, name='rec_emb')

    self.pooling = mt.layers.Pooling(FLAGS.pooling)
    self.sum_pooling = mt.layers.Pooling('sum')

    if FLAGS.use_history_emb:
      if not 'Match' in FLAGS.hpooling:
        self.hpooling = mt.layers.Pooling(FLAGS.hpooling)
      else:
        Match = getattr(mt.layers, FLAGS.hpooling)
        if FLAGS.hpooling == 'MultiHeadAttentionMatch':
          self.hpooling = Match(d_model=FLAGS.other_emb_dim, num_heads=2)
        else:
          self.hpooling = Match(combiner=None)
      if FLAGS.hpooling2:
        self.hpooling2 = mt.layers.Pooling(FLAGS.hpooling2)

      if FLAGS.history_encoder:
      #   self.history_encoder = tf.keras.layers.GRU(FLAGS.other_emb_dim, return_sequences=True)
        self.history_encoder = mt.layers.CudnnRnn(num_layers=1, 
                                  num_units=FLAGS.hidden_size, 
                                  keep_prob=1.,
                                  share_dropout=False,
                                  recurrent_dropout=False,
                                  concat_layers=True,
                                  bw_dropout=False,
                                  residual_connect=False,
                                  train_init_state=False,
                                  cell=FLAGS.history_encoder)
      else:
        self.history_encoder = lambda x, y: x
      
    # TODO position_encoding is yet another pos emb only for history
    if FLAGS.use_history_position:
      self.hpos_emb = Embedding(1000, FLAGS.other_emb_dim, name='hpos_emb')

  def call(self, input, x_doc, doc_topic_emb, doc_kw_emb):
    add, adds = self.add, self.adds
    self.clear()
    
    histories = input['vd_history'] if FLAGS.is_video else input['tw_history']
    history_id = histories
    history_topic = input["vd_history_topic"] if FLAGS.is_video else input['tw_history_topic']
    mark = 'vd' if FLAGS.is_video else 'tw'
    if not isinstance(histories, tf.Tensor):
      histories = tf.sparse.to_dense(histories, validate_indices=False)

    if not FLAGS.history_attention:
      x_docs = self.hpooling(self.doc_emb(histories), mt.length(histories))
      x_topics = self.hpooling(self.topic_emb(history_topic), mt.length(history_topic))

      adds([
          [x_docs, 'history_docs'],
          [x_topics, 'history_topics']
        ])
    else:
      if FLAGS.use_total_attn:  # mkyuwen 保留原始attn逻辑
        # query
        # x_doc, doc_kw_emb, doc_topic_emb
        doc_rec = input["rea"] if input["rea"].dtype == tf.int64 else tf.string_to_number(input["rea"], tf.int64)

        # history
        history_id = input[f"{mark}_history"]
        history_topic = input[f"{mark}_history_topic"] 
        history_kw = input[f"{mark}_history_kw"]
        history_rec = input[f"{mark}_history_rec"]

        history_kw = K.reshape(history_kw, [mt.get_shape(history_kw, 0), mt.get_shape(history_id, 1), -1])  # should be [512, length, 4]

        if FLAGS.history_length:
          history_id = history_id[:, :FLAGS.history_length]
          history_topic = history_topic[:, :FLAGS.history_length]
          history_kw = history_kw[:, :FLAGS.history_length, :]
          history_rec = history_rec[:, :FLAGS.history_length]

        # mkyuwen 0624
        history_kw_emb = self.hpooling(self.kw_emb(history_kw), axis=2)
        # # -------- mkyuwen 0624
        # if FLAGS.use_total_samekw_lbwnmktest and FLAGS.use_merge_kw_emb:  # 必须同时满足
        #     history_kw_emb = self.mktest_kw_pooling(self.mktest_user_kw_emb(history_kw), axis=2)
        # else:
        #     history_kw_emb = self.hpooling(self.kw_emb(history_kw), axis=2)
        # # --------# --------# --------# --------# --------# --------

        query = K.concatenate([x_doc, doc_kw_emb, doc_topic_emb, self.rec_emb(doc_rec)], -1)
        history = K.concatenate([self.doc_emb(history_id), history_kw_emb, self.topic_emb(history_topic), self.rec_emb(history_rec)], -1)
        hist_att = self.history_att([query, history, mt.length(history_id)])
        x_hist0, x_hist1, x_hist2, x_hist3 = tf.split(hist_att, 4, axis=1)
        adds(
          [x_hist0, x_hist1, x_hist2, x_hist3],
          ['history_doc', 'history_kw', 'history_topic', 'history_rec']
        )
      else:
        # history
        history_id = input[f"{mark}_history"]
        history_topic = input[f"{mark}_history_topic"] 

        if FLAGS.history_length:
          history_id = history_id[:, :FLAGS.history_length]
          history_topic = history_topic[:, :FLAGS.history_length]

        query = K.concatenate([x_doc, doc_topic_emb], -1)
        history = K.concatenate([self.doc_emb(history_id), self.topic_emb(history_topic)], -1)
        hist_att = self.history_att([query, history, mt.length(history_id)])
        x_hist0, x_hist1 = tf.split(hist_att, 2, axis=1)
        adds([
          [x_hist0, f'{mark}_history_doc'],
          [x_hist1, f'{mark}_history_topic']
        ])

    return self.embs


class HistoryV2(mt.Model):
  def __init__(self, doc_emb, topic_emb, kw_emb):
    super(HistoryV2, self).__init__()
  
    self.doc_emb = doc_emb
    self.topic_emb = topic_emb
    self.kw_emb = kw_emb

    self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)
    Embedding = keras.layers.Embedding
    SimpleEmbedding = mt.layers.SimpleEmbedding
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()
    kwargs = dict(num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner, 
                  embeddings_regularizer=self.regularizer, num_shards=FLAGS.num_shards)
    self.kwargs = kwargs
    self.HashEmbedding = HashEmbedding

    if FLAGS.history_attention:
      self.history_att = mt.layers.DINAttention(attention_hidden_units=FLAGS.attention_mlp_dims, attention_activation=FLAGS.attention_activation, 
                                                  mode=FLAGS.attention_mode, weight_normalization=FLAGS.attention_norm, name="din_attention")
      self.rec_emb = SimpleEmbedding(2000, FLAGS.other_emb_dim, name='rec_emb')

    self.pooling = mt.layers.Pooling(FLAGS.pooling)
    self.sum_pooling = mt.layers.Pooling('sum')

    if FLAGS.use_history_emb:
      if not 'Match' in FLAGS.hpooling:
        self.hpooling = mt.layers.Pooling(FLAGS.hpooling)
      else:
        Match = getattr(mt.layers, FLAGS.hpooling)
        if FLAGS.hpooling == 'MultiHeadAttentionMatch':
          self.hpooling = Match(d_model=FLAGS.other_emb_dim, num_heads=2)
        else:
          self.hpooling = Match(combiner=None)
      if FLAGS.hpooling2:
        self.hpooling2 = mt.layers.Pooling(FLAGS.hpooling2)

      if FLAGS.history_encoder:
      #   self.history_encoder = tf.keras.layers.GRU(FLAGS.other_emb_dim, return_sequences=True)
        self.history_encoder = mt.layers.CudnnRnn(num_layers=1, 
                                  num_units=int(FLAGS.hidden_size / 2), 
                                  keep_prob=1.,
                                  share_dropout=False,
                                  recurrent_dropout=False,
                                  concat_layers=True,
                                  bw_dropout=False,
                                  residual_connect=False,
                                  train_init_state=False,
                                  cell=FLAGS.history_encoder)
      else:
        self.history_encoder = lambda x, y: x
      
    # TODO position_encoding is yet another pos emb only for history
    if FLAGS.use_history_position:
      self.hpos_emb = Embedding(1000, FLAGS.other_emb_dim, name='hpos_emb')

  def call(self, input, x_doc, doc_topic_emb, doc_kw_emb):
    add, adds = self.add, self.adds
    histories = input['vd_history'] if FLAGS.is_video else input['tw_history']
    history_id = histories
    history_topic = input["vd_history_topic"] if FLAGS.is_video else input['tw_history_topic']
    mark = 'vd' if FLAGS.is_video else 'tw'
    mark2 = 'tw' if FLAGS.is_video else 'vd'
    if FLAGS.use_all_type:
      mark, mark2 = 'vd', 'tw'
    if not isinstance(histories, tf.Tensor):
      histories = tf.sparse.to_dense(histories, validate_indices=False)

    if not FLAGS.history_attention:
      x_docs = self.hpooling(self.doc_emb(histories), mt.length(histories))
      x_topics = self.hpooling(self.topic_emb(history_topic), mt.length(history_topic))

      adds(
        [x_docs, x_topics],
        ['history_doc', 'history_topic']
      )
    else:
      if FLAGS.use_total_attn:  # mkyuwen 保留原始attn逻辑
        # query
        # x_doc, doc_kw_emb, doc_topic_emb
        doc_rec = input["rea"] if input["rea"].dtype == tf.int64 else tf.string_to_number(input["rea"], tf.int64)

        # history
        history_id = input[f"{mark}_history"]
        history_topic = input[f"{mark}_history_topic"] 
        history_kw = input[f"{mark}_history_kw"]
        history_rec = input[f"{mark}_history_rec"]

        history_kw = K.reshape(history_kw, [mt.get_shape(history_kw, 0), mt.get_shape(history_id, 1), -1])  # should be [512, length, 4]

        if FLAGS.history_length:
          history_id = history_id[:, :FLAGS.history_length]
          history_topic = history_topic[:, :FLAGS.history_length]
          history_kw = history_kw[:, :FLAGS.history_length, :]
          history_rec = history_rec[:, :FLAGS.history_length]

        # mkyuwen 0624
        history_kw_emb = self.hpooling(self.kw_emb(history_kw), axis=2)
        # # -------- mkyuwen 0624
        # if FLAGS.use_total_samekw_lbwnmktest and FLAGS.use_merge_kw_emb:  # 必须同时满足
        #     history_kw_emb = self.mktest_kw_pooling(self.mktest_user_kw_emb(history_kw), axis=2)
        # else:
        #     history_kw_emb = self.hpooling(self.kw_emb(history_kw), axis=2)
        # # --------# --------# --------# --------# --------# --------

        query = K.concatenate([x_doc, doc_kw_emb, doc_topic_emb, self.rec_emb(doc_rec)], -1)
        history = K.concatenate([self.doc_emb(history_id), history_kw_emb, self.topic_emb(history_topic), self.rec_emb(history_rec)], -1)
        hist_att = self.history_att([query, history, mt.length(history_id)])
        x_hist0, x_hist1, x_hist2, x_hist3 = tf.split(hist_att, 4, axis=1)
        x_hist = [x_hist0, x_hist1, x_hist2, x_hist3]
        adds(
          [x_hist0, x_hist1, x_hist2, x_hist3],
          ['history_doc', 'history_kw', 'history_topic', 'history_rec']
        )
      else:
        # history
        history_id = input[f"{mark}_history"]
        history_id2 = input[f"{mark2}_history"]
        history_topic = input[f"{mark}_history_topic"] 

        if FLAGS.history_length:
          history_id = history_id[:, :FLAGS.history_length]
          history_topic = history_topic[:, :FLAGS.history_length]

        query = x_doc
        history = self.doc_emb(history_id)
        history2 = self.doc_emb(history_id2)
        history = self.history_encoder(history, mt.length(history_id))
        hist_att = self.history_att([query, history, mt.length(history_id)])
        hist_att2 = self.history_att([query, history, mt.length(history_id2)])

        x_topics = self.hpooling(self.topic_emb(history_topic), mt.length(history_topic))
        adds(
          [hist_att, hist_att2, x_topics],
          [f'{mark}_history_doc', f'{mark2}_history_doc', f'{mark}_history_topic']
          )

    return self.embs

