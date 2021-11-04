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
import re

import melt as mt
import gezi 
logging = gezi.logging

from projects.ai.mind.src.config import *
from projects.ai.mind.src import util
from projects.ai.mind.src import loss

class Encoder(keras.Model):
  def __init__(self, emb, pooling=None, emb2=None, **kwargs):
    super(Encoder, self).__init__(**kwargs)   
    self.emb = emb
    pooling = pooling or 'att'
    self.pooling = mt.layers.Pooling(pooling)
    self.output_dim = emb.output_dim

  # in eger mode ok without below but graph not
  def compute_output_shape(self, input_shape):
    return (None, self.output_dim)

  def call(self, x):
    return self.pooling(self.emb(x), mt.length(x))

class SeqsEncoder(keras.Model):
  def __init__(self, encoder, pooling=None, **kwargs):
    super(SeqsEncoder, self).__init__(**kwargs)   
    self.encoder =  keras.layers.TimeDistributed(encoder)
    pooling = pooling or 'din'
    self.pooling = util.get_att_pooling(pooling)

  def call(self, seqs, tlen, query=None):
    seqs = self.encoder(seqs)
    return self.pooling(query, seqs, tlen)

class Encoders(keras.Model):
  def __init__(self, embs, pooling=None, combiner='sum', **kwargs):
    super(Encoders, self).__init__(**kwargs)   
    self.embs = embs
    pooling = pooling or 'att'
    self.pooling = mt.layers.Pooling(pooling)
    self.output_dim = embs[0].output_dim if combiner != 'concat' else embs[0].output_dim * (len(embs))
    self.combiner = combiner
    assert self.combiner == 'sum'

  # in eger mode ok without below but graph not
  def compute_output_shape(self, input_shape):
    return (None, self.output_dim)

  def call(self, x):
    xs = tf.split(x, len(self.embs), axis=-1)
    embs = []
    for x, emb in zip(xs, self.embs):
      embs += [emb(x)]

    embs = tf.add_n(embs)

    return self.pooling(embs, mt.length(x))


class DocEncoder(keras.Model):
  pass

class HistoryEncoder(keras.Model):
  pass

class Model(keras.Model):
  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs) 
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

    self.title_lookup = mt.layers.LookupArray(FLAGS.title_lookup, name='title_lookup')
    self.doc_lookup = mt.layers.LookupArray(FLAGS.doc_lookup, name='doc_lookup')

    self.title_encoder = Encoder(self.word_emb, name='title_encoder')
    self.titles_encoder = SeqsEncoder(self.title_encoder, FLAGS.seqs_pooling, name='titles_encoder')

    self.abstract_encoder = Encoder(self.word_emb, name='abstract_encoder')
    self.abstracts_encoder = SeqsEncoder(self.abstract_encoder, name='abstracts_encoder')

    self.body_encoder = Encoder(self.word_emb, name='body_encoder')
    self.bodies_encoder = SeqsEncoder(self.body_encoder, name='bodies_encoder')

    self.entities_encoder = Encoders([self.entity_emb, self.entity_type_emb], name='entities_encoder')
    self.his_entities_encoder = SeqsEncoder(self.entities_encoder, FLAGS.seqs_pooling, name='his_entities_encoder')

    self.bert_encoder = mt.models.Bert(FLAGS.emb_size, FLAGS.bert_dir, name='bert_encoder')
    self.bert_seqs_encoder = SeqsEncoder(self.bert_encoder, name='bert_seqs_encoder')

    self.sum_pooling = mt.layers.SumPooling()
    self.mean_pooling = mt.layers.MeanPooling()
    self.pooling = mt.layers.Pooling(FLAGS.pooling)

    self.feat_pooling = mt.layers.Pooling(FLAGS.feat_pooling, name='feat_pooling')
    self.his_simple_pooling = mt.layers.Pooling(FLAGS.his_simple_pooling)
    # self.his_entity_pooling = mt.layers.Pooling('att', name='his_entity_pooling')
    self.his_entity_pooling = util.get_att_pooling('din', name='his_entity_pooling')
    self.his_cat_pooling = mt.layers.Pooling('att', name='his_cat_pooling')
    self.his_scat_din_pooling = util.get_att_pooling('din', name='his_scat_din_pooling')

    self.dense = Dense(1) if not FLAGS.use_multi_dropout else mt.layers.MultiDropout(1, drop_rate=0.3)
    self.batch_norm = BatchNormalization()
    self.dropout = keras.layers.Dropout(FLAGS.dropout)
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    mlp_dims = [FLAGS.emb_size * 2, FLAGS.emb_size] if not FLAGS.big_mlp else [FLAGS.emb_size * 4, FLAGS.emb_size * 2, FLAGS.emb_size]
    self.dense_mlp = mt.layers.MLP(mlp_dims,
                                     activation=activation, 
                                     drop_rate=FLAGS.mlp_dropout,
                                     name='dense_mlp')
    
    mlp_dims = [512, 256, 64] if not FLAGS.big_mlp else [1024, 512, 256]
    self.mlp = mt.layers.MLP(mlp_dims, activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_encoder = util.get_encoder(FLAGS.his_encoder)
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)
    self.his_pooling = util.get_att_pooling(FLAGS.his_pooling)
    self.his_pooling2 = util.get_att_pooling(FLAGS.his_pooling2)
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

    if FLAGS.his_strategy.startswith('bst'):
      self.transformer = mt.layers.transformer.Encoder(num_layers=1, d_model=FLAGS.hidden_size, num_heads=FLAGS.num_heads, 
                                                         dff=FLAGS.hidden_size, maximum_position_encoding=None, activation=FLAGS.transformer_activation,
                                                         rate=FLAGS.transformer_dropout)

    self.fusion = mt.layers.SemanticFusion(drop_rate=0.1)

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

      self.softmax_loss_function = mt.seq2seq.gen_sampled_softmax_loss_function(5,
                                                                                  vsize,
                                                                                  weights=self.sampled_weight,
                                                                                  biases=self.sampled_bias,
                                                                                  log_uniform_sample=True,
                                                                                  is_predict=False,
                                                                                  sample_seed=1234)

  def deal_dense(self, input):
    feats = []
    title_len_ = mt.scalar_feature(tf.cast(input['title_len'], tf.float32), max_val=20, scale=True)
    abstract_len_ = mt.scalar_feature(tf.cast(input['abstract_len'], tf.float32), max_val=120, scale=True)

    if FLAGS.dense_use_title_len:
      feats += [
            title_len_,
            abstract_len_,
          ]
   
    his_len_ = mt.scalar_feature(input['hist_len'], max_val=800, scale=True)

    if FLAGS.dense_use_his_len:
      feats += [
        his_len_,
      ]

    impression_len_ = mt.scalar_feature(input['impression_len'], max_val=300, scale=True)
    impression_cat_ratio_ = mt.scalar_feature(input['impression_cat_ratio'])
    impression_sub_cat_ratio_ = mt.scalar_feature(input['impression_sub_cat_ratio']) 
    if FLAGS.dense_use_impression:
      feats += [
        impression_len_,
        impression_cat_ratio_,
        impression_sub_cat_ratio_,
      ]

    if FLAGS.use_fresh:
      fresh = tf.cast(input['fresh'], tf.float32)
      fresh_hour_ = mt.scalar_feature(fresh / (3600), max_val=7 * 24, scale=True)
      fresh_day_ = mt.scalar_feature(fresh / (3600 * 12), max_val=7 * 2, scale=True)
      feats += [
        fresh_hour_,
        fresh_day_,
      ]

    if FLAGS.use_position:
      position_rel_ = mt.scalar_feature((input['position'] + 1) / (input['impression_len'] + 1))
      position_ = mt.scalar_feature((input['position'] + 1) / (300 + 1))
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
    mt.try_squeeze_dim(input)

    if not FLAGS.batch_parse:
      util.adjust(input, self.mode)
  
    self.embs = []
    self.feats = {}
      
    bs = mt.get_shape(input['did'], 0)

    def _is_ok(name):
      ok = True
      if FLAGS.incl_feats:
        ok = False
        for feat in FLAGS.incl_feats:
          if re.search(feat, name): 
            ok = True
        if not ok:
          return False
      if FLAGS.excl_feats:
        for feat in FLAGS.excl_feats:
          if re.search(feat, name):
            return False
      return True

    def _add(feat, name):
      if _is_ok(name):
        self.feats[name] = feat
        self.embs += [feat]

    def _adds(feats, names):
      for feat, name in zip(feats, names):
        _add(feat, name)

    # --------------------------  user 
    if FLAGS.use_uid:
      uemb = self.uemb(input['uid'])
      _add(uemb, 'uid')
    # --------------------------  doc
    if FLAGS.use_did:
      demb = self.demb(input['did'])
      _add(demb, 'did')

    # ---------------------------  context
    if 'history' in input:
      hlen = mt.length(input['history'])
      hlen = tf.math.maximum(hlen, 1)

    if FLAGS.use_time_emb:
      _add(self.hour_emb(input['hour']), 'hour')
      _add(self.weekday_emb(input['weekday']), 'weekday')

    if FLAGS.use_fresh_emb:
      fresh = input['fresh']
      fresh_day = tf.cast(fresh / (3600 * 12), fresh.dtype)
      fresh_hour = tf.cast(fresh / 3600, fresh.dtype)
          
      _add(self.fresh_day_emb(fresh_day), 'fresh_day')
      _add(self.fresh_hour_emb(fresh_hour), 'fresh_hour')
    
    if FLAGS.use_position_emb:
      _add(self.position_emb(input['position']), 'position')

    if FLAGS.use_history:
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
      
      _add(his_emb, 'his_id')

    if FLAGS.dev_version == 1:
      # -----------   history id
      if FLAGS.use_news_info and 'cat' in input:
        _adds(
          [
            self.cat_emb(input['cat']),
            self.scat_emb(input['sub_cat']),
            self.pooling(self.entity_type_emb(input['title_entity_types']), mt.length(input['title_entity_types'])),
            self.pooling(self.entity_type_emb(input['abstract_entity_types']), mt.length(input['abstract_entity_types']))
          ],
          ['cat', 'sub_cat', 'title_entity_types', 'abstract_entity_types']
        )
        if FLAGS.use_entities and 'title_entities' in input:
          _adds(
            [
              self.pooling(self.entity_emb(input['title_entities']), mt.length(input['title_entities'])),
              self.pooling(self.entity_emb(input['abstract_entities']), mt.length(input['abstract_entities']))
            ],
            ['title_entities', 'abstract_entities']
          )

      if FLAGS.use_history_info and 'history_cats' in input:
        _adds(
          [
            self.his_simple_pooling(self.cat_emb(input['history_cats']), mt.length(input['history_cats'])),
            self.his_simple_pooling(self.scat_emb(input['history_sub_cats']), mt.length(input['history_sub_cats'])),
          ],
          ['history_cats', 'history_sub_cats']
        )
        if FLAGS.use_history_entities:
          _adds(
            [
              self.his_simple_pooling(self.entity_type_emb(input['history_title_entity_types']), mt.length(input['history_title_entity_types'])),
              self.his_simple_pooling(self.entity_type_emb(input['history_abstract_entity_types']), mt.length(input['history_abstract_entity_types']))
            ],
            ['history_title_entity_types', 'history_abstract_entity_types']
          )
          if FLAGS.use_entities and 'title_entities' in input:
            _adds(
              [
                self.his_simple_pooling(self.entity_emb(input['history_title_entities']), mt.length(input['history_title_entities'])),
                self.his_simple_pooling(self.entity_emb(input['history_abstract_entities']), mt.length(input['history_abstract_entities']))
              ],
              ['history_title_entities', 'history_abstract_entities']
            )

      if FLAGS.use_title:
        cur_title = self.title_encoder(self.title_lookup(input['ori_did']))
        dids = input['ori_history']
        if FLAGS.max_titles:
          dids = dids[:, :FLAGS.max_titles]
        # TODO should hlen be mt.length(dids)
        his_title = self.titles_encoder(self.title_lookup(dids), tf.math.minimum(hlen, FLAGS.max_titles), cur_title)
        _adds(
          [
          cur_title,
          his_title
          ],
          ['cur_title', 'his_title']
        )
    else:
      # --------------- doc info
      doc_feats = gezi.get('doc_feats')
      doc_feat_lens = gezi.get('doc_feat_lens')
      doc = mt.lookup_feats(input['ori_did'], self.doc_lookup, doc_feats, doc_feat_lens)

      # cat = tf.squeeze(doc['cat'], -1)
      # sub_cat = tf.squeeze(doc['sub_cat'], -1)

      # # TODO get_id起作用 但是 embeding仍然必须是比较大的 按get id后最大的情况 缩小之后报错
      # title_entities = doc['title_entities']
      # title_entity_types = doc['title_entity_types']
      # abstract_entities = doc['abstract_entities']
      # abstract_entity_types = doc['abstract_entity_types']

      cat = input['cat']
      sub_cat = input['sub_cat']
      title_entities = input['title_entities']
      title_entity_types = input['title_entity_types']
      abstract_entities = input['abstract_entities']
      abstract_entity_types = input['abstract_entity_types']
    
      # mt.length 不用速度会慢
      title_entities = self.entities_encoder(tf.concat([title_entities, title_entity_types], -1))
      abstract_entities = self.entities_encoder(tf.concat([abstract_entities, abstract_entity_types], -1))
      # prev_cat_emb = self.cat_emb(cat)
      # prev_scat_emb = self.scat_emb(cat)
      cat_emb = self.cat_emb(cat)
      scat_emb = self.scat_emb(sub_cat)
      _adds(
        [
          # prev_cat_emb,
          # prev_scat_emb,
          cat_emb,
          scat_emb,
          # self.pooling(self.entity_emb(title_entities), mt.length(doc['title_entities'])),
          # self.pooling(self.entity_type_emb(title_entity_types), mt.length(doc['title_entity_types'])),
          # self.pooling(self.entity_emb(abstract_entities), mt.length(doc['abstract_entities'])),
          # self.pooling(self.entity_type_emb(abstract_entity_types), mt.length(doc['abstract_entity_types'])),
          title_entities,
          abstract_entities
        ],
        # ['cat', 'sub_cat', 'title_entity_types', 'abstract_entity_types', 'title_entities', 'abstract_entities']
        [
          # 'prev_cat', 'prev_scat', 
          'cat', 'sub_cat', 'title_entities', 'abstract_entities'
          ]
      )

      # _adds(
      #     [
      #       self.his_simple_pooling(self.entity_type_emb(input['history_title_entity_types']), mt.length(input['history_title_entity_types'])),
      #       self.his_simple_pooling(self.entity_type_emb(input['history_abstract_entity_types']), mt.length(input['history_abstract_entity_types']))
      #     ],
      #     ['history_title_entity_merge_types', 'history_abstract_entity_merge_types']
      # )
      input['history_title_entities'] = input['history_title_entities'][:,:FLAGS.max_his_title_entities * FLAGS.max_lookup_history]
      input['history_title_entity_types'] = input['history_title_entity_types'][:,:FLAGS.max_his_title_entities * FLAGS.max_lookup_history]
      input['history_abstract_entities'] = input['history_abstract_entities'][:,:FLAGS.max_his_title_entities * FLAGS.max_lookup_history]
      input['history_abstract_entity_types'] = input['history_abstract_entity_types'][:,:FLAGS.max_his_title_entities * FLAGS.max_lookup_history]
      _adds(
          [
            self.his_entity_pooling(title_entities, (self.entity_emb(input['history_title_entities']) + self.entity_type_emb(input['history_title_entity_types'])), mt.length(input['history_title_entities'])),
            self.his_entity_pooling(abstract_entities, (self.entity_emb(input['history_abstract_entities']) + self.entity_type_emb(input['history_abstract_entity_types'])), mt.length(input['history_abstract_entities']))
          ],
          ['history_title_merge_entities', 'history_abstract_merge_entities']
      )     

      # --------------- history info
      dids = input['ori_history']
      dids = dids[:,:FLAGS.max_lookup_history]
      hlen = mt.length(input['history'])
      hlen = tf.math.maximum(hlen, 1)

      his = mt.lookup_feats(dids, self.doc_lookup, doc_feats, doc_feat_lens)

      his_cats = his['cat']
      his_cats = tf.squeeze(his_cats, -1)
      his_sub_cats = his['sub_cat']
      his_sub_cats = tf.squeeze(his_sub_cats, -1)

      his_title_entities = his['title_entities']
      his_title_entity_types = his['title_entity_types']
      his_abstract_entities = his['abstract_entities']
      his_abstract_entity_types = his['abstract_entity_types']

      # his_title_entities = self.his_entities_encoder(tf.concat([his_title_entities, his_title_entity_types], -1), 
      #                                                tf.math.minimum(hlen, FLAGS.max_titles), title_entities)
      # his_abstract_entities = self.his_entities_encoder(tf.concat([his_abstract_entities, his_abstract_entity_types], -1), 
      #                                                   tf.math.minimum(hlen, FLAGS.max_titles), abstract_entities)

      # FIXME 当前如果直接展平 mt.length有问题 因为都是内壁 0 pad,  类似  2,3,0,0 1,0,0,0  会丢掉很多信息 填1 是一种方式 （v1就是这种 最多 1,1)
      # 另外也可以用encoder
      _adds(
        [
          self.his_cat_pooling(self.cat_emb(his_cats), mt.length(his_cats)),
          self.his_cat_pooling(self.scat_emb(his_sub_cats), mt.length(his_sub_cats)),
          ## 对应cat din效果不如att(增加也没有收益) 对应title din效果比att好, entity也是din比较好
          # self.his_scat_din_pooling(scat_emb, self.scat_emb(his_sub_cats), mt.length(his_sub_cats)),
          # his_title_entities,
          # his_abstract_entities,
        ],
        [
         'history_cats', 'history_sub_cats',
        #  'history_title_entities', 'history_abstract_entities'
         ]
      )

      cur_title = self.title_encoder(doc['title'])
      his_titles = his['title']
      if FLAGS.max_titles:
        his_titles = his_titles[:, :FLAGS.max_titles]
      his_title = self.titles_encoder(his_titles, tf.math.minimum(hlen, FLAGS.max_titles), cur_title)
      _adds(
        [
        cur_title,
        his_title
        ],
        ['cur_title', 'his_title']
      )

      cur_abstract = self.abstract_encoder(doc['abstract'])
      his_abstracts = his['abstract']
      if FLAGS.max_abstracts:
        his_abstracts = his_abstracts[:, :FLAGS.max_abstracts]
      his_abstract = self.abstracts_encoder(his_abstracts, tf.math.minimum(hlen, FLAGS.max_abstracts), cur_abstract)
      _adds(
        [
        cur_abstract,
        his_abstract
        ],
        ['cur_abstract', 'his_abstract']
      )

      if FLAGS.bert_dir:
        bert_title = self.bert_encoder(doc['title_uncased'])
        # max_titles = int(FLAGS.max_titles / 5)
        max_titles = 5
        his_bert_titles = self.bert_seqs_encoder(his['title_uncased'][:, :max_titles], tf.math.minimum(hlen, max_titles), bert_title)
        _adds(
                [
                  bert_title,
                  his_bert_titles,
                ],
                [
                  'bert_title', 
                  'his_bert_titles'
                  ]
              )     

      if FLAGS.use_body:
        cur_body = self.body_encoder(doc['body'])
        his_bodies = his['body']
        if FLAGS.max_bodies:
          his_bodies = his_bodies[:, :FLAGS.max_abstracts]
        his_body = self.bodies_encoder(his_bodies, tf.math.minimum(hlen, FLAGS.max_bodies), cur_body)
        _adds(
          [
           cur_body,
           his_body,
          ],
          ['cur_body', 'his_body']
        )

    # 用impression id 会dev test不一致 不直接用id
    if FLAGS.use_impressions:
      _add(self.mean_pooling(self.demb(input['impressions'])), 'impressions')

    if FLAGS.use_dense:
      dense_emb = self.deal_dense(input)
      _add(dense_emb, 'dense')
    
    embs = self.embs
    logging.info('-----------embs:', len(embs))
    logging.info(self.feats.keys())
    logging.debug(self.feats)
    embs = [x if len(mt.get_shape(x)) == 2 else tf.squeeze(x, 1) for x in embs]
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
