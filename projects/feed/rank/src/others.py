#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   others.py
#        \author   chenghuige  
#          \date   2020-07-22 15:11:13.136206
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
from projects.feed.rank.src.history import *
from projects.feed.rank.src import history
from projects.feed.rank.src.keywords import *

import gezi 
logging = gezi.logging

class Others(mt.Model):
  def __init__(self):
    super(Others, self).__init__()

    # self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)
    self.regularizer = None
    Embedding = keras.layers.Embedding
    SimpleEmbedding = mt.layers.SimpleEmbedding
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()
    kwargs = dict(num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner, 
                  embeddings_regularizer=self.regularizer, num_shards=FLAGS.num_shards)
    self.kwargs = kwargs
    self.HashEmbedding = HashEmbedding

    if FLAGS.use_user_emb:
      self.user_emb = HashEmbeddingUD(int(FLAGS.feature_dict_size * FLAGS.user_emb_factor), FLAGS.other_emb_dim, name='user_emb', **kwargs)

    if FLAGS.use_doc_emb:
      self.doc_emb = HashEmbeddingUD(int(FLAGS.feature_dict_size * FLAGS.doc_emb_factor), FLAGS.other_emb_dim, name='doc_emb', **kwargs)

    self.topic_emb = None
    self.kw_emb = None

    self.mktest_kw_emb = SimpleEmbedding(FLAGS.keyword_dict_size, FLAGS.other_emb_dim, name='mktest_kw_emb')
    # mkyuwen kw
    if FLAGS.use_merge_kw_emb:
      if not FLAGS.use_w2v_kw_emb:
        # self.mktest_kw_emb = SimpleEmbedding(FLAGS.keyword_dict_size, FLAGS.other_emb_dim, name='mktest_kw_emb')
        # 这部分统一，下面的好写一些
        self.mktest_user_kw_emb = self.mktest_kw_emb
        self.mktest_doc_kw_emb = self.mktest_kw_emb
      else:
        # mkyuwen 0612
        # https://www.cnblogs.com/weiyinfu/p/9873001.html
        ini_w2v_emb_weights, ini_user_w2v_emb_weights, ini_doc_w2v_emb_weights = util.load_pretrained_w2v_emb()

        if not FLAGS.use_split_w2v_kw_emb:  # 公用kw-emb
          # self.mktest_kw_emb = keras.layers.Embedding(input_dim=FLAGS.keyword_dict_size + 1, output_dim=FLAGS.other_emb_dim,
          #                                             weights=[ini_w2v_emb_weights.reshape(FLAGS.keyword_dict_size + 1, FLAGS.other_emb_dim)], 
          #                                             name='mktest_0_w2v_kw_emb')  # not work
          self.mktest_kw_emb = keras.layers.Embedding(input_dim=FLAGS.keyword_dict_size + 1, output_dim=FLAGS.other_emb_dim,
                                                      embeddings_initializer=keras.initializers.constant(ini_w2v_emb_weights.reshape((FLAGS.keyword_dict_size + 1, FLAGS.other_emb_dim))), 
                                                      # trainable=trainable_,
                                                      name='mktest_uniform_UnD_w2v_kw_emb')
          # 这部分统一，下面的好写一些
          self.mktest_user_kw_emb = self.mktest_kw_emb
          self.mktest_doc_kw_emb = self.mktest_kw_emb
        else:  # doc/user分开kw_emb
          self.mktest_user_kw_emb = keras.layers.Embedding(input_dim=FLAGS.keyword_dict_size + 1, output_dim=FLAGS.other_emb_dim,
                                                      embeddings_initializer=keras.initializers.constant(ini_user_w2v_emb_weights.reshape((FLAGS.keyword_dict_size + 1, FLAGS.other_emb_dim))), 
                                                      # trainable=trainable_,
                                                      name='mktest_uniform_User_w2v_kw_emb')
          self.mktest_doc_kw_emb = keras.layers.Embedding(input_dim=FLAGS.keyword_dict_size + 1, output_dim=FLAGS.other_emb_dim,
                                                      embeddings_initializer=keras.initializers.constant(ini_doc_w2v_emb_weights.reshape((FLAGS.keyword_dict_size + 1, FLAGS.other_emb_dim))), 
                                                      # trainable=trainable_,
                                                      name='mktest_uniform_Doc_w2v_kw_emb')

    if FLAGS.use_kw_emb or FLAGS.history_attention:
      # TODO
      # self.kw_emb = SimpleEmbedding(FLAGS.keyword_dict_size, FLAGS.other_emb_dim, name='kw_emb')
      self.kw_emb = self.mktest_kw_emb
        
    if FLAGS.use_topic_emb or FLAGS.history_attention:
      # 10000
      self.topic_emb = SimpleEmbedding(FLAGS.topic_dict_size, FLAGS.other_emb_dim, name='topic_emb')
      # self.topic_emb = self.kw_emb

    if FLAGS.use_time_emb:
      self.time_emb = Embedding(500, FLAGS.other_emb_dim, name='time_emb')
      self.weekday_emb = Embedding(10, FLAGS.other_emb_dim, name='weekday_emb')

    if FLAGS.use_timespan_emb:
      self.timespan_emb = Embedding(300, FLAGS.other_emb_dim, name='timespan_emb')

    if FLAGS.use_deep_position_emb:
      self.pos_emb = Embedding(FLAGS.num_positions, FLAGS.other_emb_dim, name='pos_emb')

    if FLAGS.use_product_emb:
      self.product_emb = Embedding(10, FLAGS.other_emb_dim, name='product_emb')

    if FLAGS.use_cold_emb:
      self.cold_emb = Embedding(10, FLAGS.other_emb_dim, name='cold_emb')
    
    if FLAGS.use_title_emb:
      self.title_emb = SimpleEmbedding(100000, FLAGS.other_emb_dim, name='title_emb') if not FLAGS.title_share_kw_emb else self.kw_emb
      if FLAGS.title_encoder in ['gru', 'lstm']:
        return_sequences = FLAGS.title_pooling is not None
        # self.title_encoder = tf.keras.layers.GRU(FLAGS.other_emb_dim, return_sequences=return_sequences, 
        #                                           dropout=FLAGS.title_drop, recurrent_dropout=FLAGS.title_drop_rec)
        self.title_encoder = mt.layers.CudnnRnn(num_layers=1, 
                                  num_units=int(FLAGS.hidden_size / 2), 
                                  keep_prob=1.,
                                  share_dropout=False,
                                  recurrent_dropout=False,
                                  concat_layers=True,
                                  bw_dropout=False,
                                  residual_connect=False,
                                  train_init_state=False,
                                  cell=FLAGS.title_encoder)
      else:
        self.title_encoder  = lambda x, y: x
      if FLAGS.title_pooling:
        self.title_pooling = mt.layers.Pooling(FLAGS.title_pooling)
    
    if FLAGS.use_refresh_emb:
      self.refresh_coldstart_emb = Embedding(1001, FLAGS.other_emb_dim, name='refresh_coldstart_emb')
      self.refresh_today_emb = Embedding(1001, FLAGS.other_emb_dim, name='refresh_today_emb')

    # mkyuwen
    if FLAGS.use_distribution_emb:  # 0430
      self.distribution_emb = Embedding(1000, FLAGS.other_emb_dim, name='distribution_id_emb')

    if FLAGS.use_network_emb:
      self.network_emb = Embedding(10, FLAGS.other_emb_dim, name='network_emb')

    if FLAGS.use_activity_emb:
      self.activity_emb = Embedding(10, FLAGS.other_emb_dim, name='activity_emb')   

    if FLAGS.use_type_emb:
      self.type_emb = Embedding(10, FLAGS.other_emb_dim, name='type_emb')    

    self.pooling = mt.layers.Pooling(FLAGS.pooling)
    self.sum_pooling = mt.layers.Pooling('sum')

    if FLAGS.use_history_emb:
      HistoryEncoder = getattr(history, 'History' + FLAGS.history_strategy)
      self.history_encoder = HistoryEncoder(self.doc_emb, self.topic_emb, self.kw_emb)

  def call(self, input):
    add, adds = self.add, self.adds
    self.clear()
    
    if FLAGS.use_user_emb:
      with mt.device(FLAGS.emb_device):
        x_user = self.user_emb(input['uid'])
      self.x_user = x_user
      add(x_user, 'uid')      

    if FLAGS.use_doc_emb:
      with mt.device(FLAGS.emb_device):
        x_doc = self.doc_emb(input['did'])
      x_doc_ = x_doc
      self.x_doc = x_doc
      add(x_doc, 'did')

    # mkyuwen 0624 mv down[1]
    if FLAGS.use_kw_emb or FLAGS.history_attention:
      doc_kw = input["doc_keyword"] 
      doc_kw_emb = self.pooling(self.kw_emb(doc_kw), mt.length(doc_kw))

      self.doc_kw_emb = doc_kw_emb
      
    if FLAGS.use_topic_emb or FLAGS.history_attention:
      doc_topic_emb = self.topic_emb(input['doc_topic'])
      self.doc_topic_emb = doc_topic_emb

    if FLAGS.use_topic_emb:
      add(doc_topic_emb, 'doc_topic')
    # mkyuwen 0624 mv down[1]
    if FLAGS.use_kw_emb:
      add(doc_kw_emb, 'doc_kw')

    # mkyuwen kw
    # ---------- mkyuwen 0504
    if FLAGS.use_merge_kw_emb:
      mktest_tw_history_kw = input['mktest_tw_history_kw_feed']
      mktest_vd_history_kw = input['mktest_vd_history_kw_feed']  # 0521
      mktest_rel_vd_history_kw = input['mktest_rel_vd_history_kw_feed'] 
      mktest_doc_kw = input['mktest_doc_kw_feed']
      mktest_doc_kw_secondary = input['mktest_doc_kw_secondary_feed']
      mktest_tw_long_term_kw = input['mktest_tw_long_term_kw_feed']
      mktest_vd_long_term_kw = input['mktest_vd_long_term_kw_feed'] 
      mktest_new_search_kw = input['mktest_new_search_kw_feed']
      mktest_long_search_kw = input['mktest_long_search_kw_feed']
      mktest_user_kw = input['mktest_user_kw_feed'] 
      if FLAGS.use_w2v_kw_emb:  # mkyuwen w2v 参考x = (x + 1) * mask, 做emb的时候已经word_index = hash mod 100w +1
        # ------------ case1：只有非0的index+1,0保持不变。与mt.length逻辑保持一致
        mktest_tw_history_kw_mask0 = tf.cast(mktest_tw_history_kw > 0, tf.int64)  # >0的非padding部分，为1
        mktest_tw_history_kw = (mktest_tw_history_kw + mktest_tw_history_kw_mask0)  # use_w2v_kw_emb, index+1 手动
        mktest_vd_history_kw = (mktest_vd_history_kw + tf.cast(mktest_vd_history_kw > 0, tf.int64))  # 
        mktest_rel_vd_history_kw = (mktest_rel_vd_history_kw + tf.cast(mktest_rel_vd_history_kw > 0, tf.int64))  # 
        mktest_doc_kw = (mktest_doc_kw + tf.cast(mktest_doc_kw > 0, tf.int64))  # 
        mktest_doc_kw_secondary = (mktest_doc_kw_secondary + tf.cast(mktest_doc_kw_secondary > 0, tf.int64))  # 
        mktest_tw_long_term_kw = (mktest_tw_long_term_kw + tf.cast(mktest_tw_long_term_kw > 0, tf.int64))  # 
        mktest_vd_long_term_kw = (mktest_vd_long_term_kw + tf.cast(mktest_vd_long_term_kw > 0, tf.int64))  # 
        mktest_new_search_kw = (mktest_new_search_kw + tf.cast(mktest_new_search_kw > 0, tf.int64))  # 
        mktest_long_search_kw = (mktest_long_search_kw + tf.cast(mktest_long_search_kw > 0, tf.int64))  # 
        mktest_user_kw = (mktest_user_kw + tf.cast(mktest_user_kw > 0, tf.int64))  # 
      
      # mkyuwen 0612 
      # 分开user/doc, 两者内容是否公用一个在上面判断，对这里透明
      # self.sum_pooling = mt.layers.Pooling('sum')
      self.mktest_kw_pooling = mt.layers.Pooling(FLAGS.merge_kw_emb_pooling)
      if FLAGS.use_tw_history_kw_merge_emb: # user
        add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_tw_history_kw), mt.length(mktest_tw_history_kw)), 'mktest_tw_history_kw')
      if FLAGS.use_vd_history_kw_merge_emb: # user
        add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_vd_history_kw), mt.length(mktest_vd_history_kw)), 'mktest_vd_history_kw')
      if FLAGS.use_rel_vd_history_kw_merge_emb: # user
        add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_rel_vd_history_kw), mt.length(mktest_rel_vd_history_kw)), 'mktest_vd_history_kw')
      if FLAGS.use_doc_kw_merge_emb: # doc
        if FLAGS.use_kw_merge_score:
          mktest_doc_kw_ = self.mktest_doc_kw_emb(mktest_doc_kw)
          mktest_doc_kw_score = input['mktest_doc_kw_score_feed'] 
          mktest_doc_kw_score = K.expand_dims(mktest_doc_kw_score, -1)
          # print ("mktest check mktest_doc_kw_",mktest_doc_kw_.get_shape())
          # print ("mktest check mktest_doc_kw_score",mktest_doc_kw_score.get_shape())
          add(self.mktest_kw_pooling(mktest_doc_kw_ * mktest_doc_kw_score, mt.length(mktest_doc_kw)), 'mktest_doc_kw')
        else:
          add(self.mktest_kw_pooling(self.mktest_doc_kw_emb(mktest_doc_kw), mt.length(mktest_doc_kw)), 'mktest_doc_kw')
      if FLAGS.use_doc_kw_secondary_merge_emb:  # doc
        if FLAGS.use_kw_secondary_merge_score:
          mktest_doc_kw_secondary_ = self.mktest_doc_kw_emb(mktest_doc_kw_secondary)
          mktest_doc_kw_secondary_score = input['mktest_doc_kw_secondary_score_feed'] 
          mktest_doc_kw_secondary_score = K.expand_dims(mktest_doc_kw_secondary_score, -1)
          add(self.mktest_kw_pooling(mktest_doc_kw_secondary_ * mktest_doc_kw_secondary_score, mt.length(mktest_doc_kw_secondary)), 'mktest_doc_kw_secondary')
        else:
          add(self.mktest_kw_pooling(self.mktest_doc_kw_emb(mktest_doc_kw_secondary), mt.length(mktest_doc_kw_secondary)), 'mktest_doc_kw_secondary')
      if FLAGS.use_tw_long_term_kw_merge_emb: # user
        add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_tw_long_term_kw), mt.length(mktest_tw_long_term_kw)), 'mktest_tw_long_term_kw')
      if FLAGS.use_vd_long_term_kw_merge_emb:  # user
        add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_vd_long_term_kw), mt.length(mktest_vd_long_term_kw)), 'mktest_vd_long_term_kw')
      if FLAGS.use_new_search_kw_merge_emb:  # user
        if FLAGS.use_new_search_kw_merge_score:
          mktest_new_search_kw_ = self.mktest_user_kw_emb(mktest_new_search_kw)
          mktest_new_search_kw_score = input['mktest_new_search_kw_score_feed'] 
          mktest_new_search_kw_score = K.expand_dims(mktest_new_search_kw_score, -1)
          add(self.mktest_kw_pooling(mktest_new_search_kw_ * mktest_new_search_kw_score, mt.length(mktest_new_search_kw)), 'mktest_new_search_kw')
        else:
          add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_new_search_kw), mt.length(mktest_new_search_kw)), 'mktest_new_search_kw')
      if FLAGS.use_long_search_kw_merge_emb: # user
        add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_long_search_kw), mt.length(mktest_long_search_kw)), 'mktest_long_search_kw')
      if FLAGS.use_user_kw_merge_emb: # user
        if FLAGS.use_user_kw_merge_score:
          mktest_user_kw_ = self.mktest_user_kw_emb(mktest_user_kw)
          mktest_user_kw_score = input['mktest_user_kw_score_feed'] 
          mktest_user_kw_score = K.expand_dims(mktest_user_kw_score, -1)
          add(self.mktest_kw_pooling(mktest_user_kw_ * mktest_user_kw_score, mt.length(mktest_user_kw)), 'mktest_user_kw')
        else:
          add(self.mktest_kw_pooling(self.mktest_user_kw_emb(mktest_user_kw), mt.length(mktest_user_kw)), 'mktest_user_kw')
      
      # other_embs += [(other_embs[-1] + other_embs[-2] + other_embs[-3]) / 3.]
    # ----------# ----------# ----------# ----------# ----------# ----------

    # # mkyuwen 0624 mv [1] here
    # if FLAGS.use_topic_emb or FLAGS.history_attention:
    #   doc_kw = input["doc_keyword"] % FLAGS.keyword_dict_size
    #   # doc_kw_emb = self.pooling(self.kw_emb(doc_kw), mt.length(doc_kw))
    #   # -------- mkyuwen 0624
    #   if FLAGS.use_total_samekw_lbwnmktest and FLAGS.use_merge_kw_emb:  # 必须同时满足
    #     doc_kw_emb = self.mktest_kw_pooling(self.mktest_doc_kw_emb(doc_kw), mt.length(doc_kw))
    #   else:
    #     doc_kw_emb = self.pooling(self.kw_emb(doc_kw), mt.length(doc_kw))
      
    # # mkyuwen 0624 mv [1] here
    # if FLAGS.use_kw_emb:
    #   other_embs += [doc_kw_emb]
    # --------# --------# --------# --------# --------# --------

    if FLAGS.use_history_emb:
      self.history_encoder(input, x_doc, doc_topic_emb, doc_kw_emb)
      self.merge(self.history_encoder.feats)

    if FLAGS.use_time_emb:
      if FLAGS.use_time_so:
        time_module = tf.load_op_library('./ops/time.so')
        get_time_intervals = time_module.time
      else:
        def get_time_intervals(x):
          res = tf.numpy_function(util.get_time_intervals, [x], x.dtype)
          res.set_shape(x.get_shape())
          return res

      time_interval = input['time_interval']
      if FLAGS.time_smoothing:
        x_time = self.time_emb(time_interval)
        num_bins = FLAGS.time_bins_per_hour * 24
        tmask = tf.cast(time_interval > 1, x_time.dtype)
        tbase = time_interval * (1 - tmask)
        time_pre = (time_interval - 2 -1 * FLAGS.time_bins_per_hour) % num_bins + 2 
        time_pre = tbase + time_pre * tmask
        time_pre2 = (time_interval - 2 -2 * FLAGS.time_bins_per_hour) % num_bins + 2
        time_pre2 = tbase + time_pre2 * tmask
        time_after = (time_interval - 2 + 1 * FLAGS.time_bins_per_hour) % num_bins + 2
        time_after = tbase + time_after * tmask
        time_after2 = (time_interval - 2 + 2 * FLAGS.time_bins_per_hour) % num_bins + 2
        time_after2 = tbase + time_after2 * tmask
        x_time_pre = self.time_emb(time_pre)
        x_time_pre2 = self.time_emb(time_pre2)
        x_time_after = self.time_emb(time_after)
        x_time_after2 = self.time_emb(time_after2)
        x_time = (0.4 * x_time + 0.2 * x_time_pre + 0.1 * x_time_pre2 + 0.2 * x_time_after + 0.1 * x_time_after2) / 5.
      # print('x_time2', x_time)
      elif FLAGS.time_bins_per_day:
        num_bins = FLAGS.time_bins_per_hour * 24
        num_large_bins = FLAGS.time_bins_per_day
        intervals_per_large_bin = tf.cast(num_bins / num_large_bins, time_interval.dtype)
        tmask = tf.cast(time_interval > 1, time_interval.dtype)
        tbase = time_interval * (1 - tmask)
        time_interval_large = tf.cast(((time_interval - 2 - FLAGS.time_bin_shift_hours * FLAGS.time_bins_per_hour) % num_bins)/ intervals_per_large_bin, time_interval.dtype) + 2
        time_interval_large = tbase + time_interval_large * tmask
        x_time = self.time_emb(time_interval_large)
      else:
        x_time = self.time_emb(time_interval)

      time_weekday = input['time_weekday'] 
      x_weekday = self.weekday_emb(time_weekday)
    
      adds([
        [x_time, 'time'],
        [x_weekday, 'weekday']
      ])

      # if FLAGS.use_dense_feats:
      #   # TODO remove dense feats of time as 23 and 00... 
      #   s_time = tf.cast(time_interval, tf.float32) / (24 * FLAGS.time_bins_per_hour + 10.)
      #   s_time = tf.zeros_like(time_interval)
      #   s_time = mt.youtube_scalar_features(s_time)

      #   s_weekday = tf.cast(time_weekday, tf.float32) / 10.
      #   s_weekday = mt.youtube_scalar_features(s_weekday)

        # dense_feats = [s_time, s_weekday]

    if FLAGS.use_timespan_emb:
      if FLAGS.use_time_so:
        get_timespan_intervals = time_module.timespan
      else:
        def get_timespan_intervals(x, y): 
          res = tf.numpy_function(util.get_timespan_intervals, [x, y], x.dtype)
          res.set_shape(x.get_shape())
          return res

      timespan_interval = input['timespan_interval']
      x_timespan = self.timespan_emb(timespan_interval)
      add(x_timespan, 'timespan')

      # if FLAGS.use_dense_feats:
      #   s_timespan = tf.cast(timespan_interval, tf.float32) / 200. 
      #   s_timespan = mt.youtube_scalar_features(s_timespan)
      #   # dense_feats += [s_timespan]

      #   s_timespan2 = input['impression_time'] - input['article_page_time']
      #   max_delta = 3000000
      #   s_timespan2 = tf.math.minimum(s_timespan2, max_delta)
      #   # s_timespan2 = tf.math.maximum(s_timespan2, -10)
      #   s_timespan2 = tf.math.maximum(s_timespan2, 0)
      #   s_timespan2 = tf.cast(s_timespan2, tf.float32) / float(max_delta)
      #   s_timespan2 = mt.youtube_scalar_features(s_timespan2)
        # dense_feats += [s_timespan2]

    if FLAGS.use_product_emb:
      x_product = self.product_emb(util.get_product_id(input['product']))
      add(x_product, 'product')

    if FLAGS.use_cold_emb:
      cold = input['cold'] if not FLAGS.is_infer else tf.cast(util.is_cb_user(input['rea']), input['index'].dtype)
      x_cold = self.cold_emb(cold) 
      add(x_cold, 'cold')

    if FLAGS.use_title_emb:
      x_title = self.title_emb(input['title'])
      # x_title = self.title_encoder(x_title, mask=tf.sequence_mask(mt.length(input['title'])))
      x_title = self.title_encoder(x_title, mt.length(input['title']))
      x_title = self.title_pooling(x_title, mt.length(input['title']))
      add(x_title, 'title')

    if FLAGS.use_refresh_emb:
      x_refresh1 = self.refresh_coldstart_emb(tf.math.minimum(input['coldstart_refresh_num'], 1000))
      x_refresh2 = self.refresh_today_emb(tf.math.minimum(input['today_refresh_num'], 1000))
      adds([
        [x_refresh1, 'coldstart_refresh'],
        [x_refresh2, 'today_refresh']
      ])

    # mkyuwen 0520(本身input自带feed)
    if FLAGS.use_distribution_emb:  # 0430
      input['mktest_distribution_id_feed'] = input['mktest_distribution_id_feed'] % 1000
      x_disid = self.distribution_emb(input['mktest_distribution_id_feed'])
      add(x_disid, 'distribution')

    if FLAGS.use_network_emb:
      x_network = self.network_emb(input['network'])
      add(x_network, 'network')

    if FLAGS.use_activity_emb:
      x_activity = self.activity_emb(input['user_active'] + 1)
      add(x_activity, 'user_active')

    if FLAGS.use_type_emb:
      x_type = self.type_emb(input['type'])
      add(x_type, 'type')

    other_embs = self.embs
    other_embs = [x if len(x.get_shape()) == 2 else tf.squeeze(x, 1) for x in other_embs]
    return other_embs

  def init_predict(self, input, dummy):
    # if FLAGS.use_user_emb:
    input['uid'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'uid_feed')
    tf.compat.v1.add_to_collection('uid_feed', input['uid'])
    input['uid'] += dummy

    # if FLAGS.use_doc_emb:
    input['did'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'did_feed')
    tf.compat.v1.add_to_collection('did_feed', input['did'])
    input['did'] += dummy

    if FLAGS.use_title_emb:
      title_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'title_feed')
      tf.compat.v1.add_to_collection('title_feed', title_feed)
      input['title'] = title_feed

    input['history'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'doc_idx_feed')
    tf.compat.v1.add_to_collection('doc_idx_feed', input['history'])
    input['history'] += dummy

    input['keyword'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'kw_idx_feed')
    tf.compat.v1.add_to_collection('kw_idx_feed', input['keyword'])
    input['keyword'] += dummy

    input['topic'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'topic_idx_feed')
    tf.compat.v1.add_to_collection('topic_idx_feed', input['topic'])
    input['topic'] += dummy

    input['doc_keyword'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None],'doc_kw_idx_feed')
    tf.compat.v1.add_to_collection('doc_kw_idx_feed', input['doc_keyword'])
    input['doc_keyword'] += dummy

    input['doc_topic'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'doc_topic_idx_feed')
    tf.compat.v1.add_to_collection('doc_topic_idx_feed', input['doc_topic'])
    input['doc_topic'] += dummy

    input['impression_time'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'time_feed')
    tf.compat.v1.add_to_collection('time_feed', input['impression_time'])
    input['impression_time'] += dummy
    input['impression_time'] = tf.squeeze(input['impression_time'], 1)

    input['article_page_time'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'ptime_feed')
    tf.compat.v1.add_to_collection('ptime_feed', input['article_page_time'])
    input['article_page_time'] += dummy
    input['article_page_time'] = tf.squeeze(input['article_page_time'], 1)

    # mkyuwen
    input['mktest_tw_history_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_tw_history_kw_feed')
    tf.compat.v1.add_to_collection('mktest_tw_history_kw_feed', input['mktest_tw_history_kw_feed'])
    input['mktest_tw_history_kw_feed'] += dummy

    input['mktest_vd_history_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_vd_history_kw_feed')
    tf.compat.v1.add_to_collection('mktest_vd_history_kw_feed', input['mktest_vd_history_kw_feed'])
    input['mktest_vd_history_kw_feed'] += dummy

    input['mktest_rel_vd_history_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_rel_vd_history_kw_feed')
    tf.compat.v1.add_to_collection('mktest_rel_vd_history_kw_feed', input['mktest_rel_vd_history_kw_feed'])
    input['mktest_rel_vd_history_kw_feed'] += dummy

    input['mktest_doc_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_doc_kw_feed')
    tf.compat.v1.add_to_collection('mktest_doc_kw_feed', input['mktest_doc_kw_feed'])
    input['mktest_doc_kw_feed'] += dummy

    input['mktest_doc_kw_secondary_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_doc_kw_secondary_feed')
    tf.compat.v1.add_to_collection('mktest_doc_kw_secondary_feed', input['mktest_doc_kw_secondary_feed'])
    input['mktest_doc_kw_secondary_feed'] += dummy

    input['mktest_tw_long_term_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_tw_long_term_kw_feed')
    tf.compat.v1.add_to_collection('mktest_tw_long_term_kw_feed', input['mktest_tw_long_term_kw_feed'])
    input['mktest_tw_long_term_kw_feed'] += dummy

    input['mktest_vd_long_term_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_vd_long_term_kw_feed')
    tf.compat.v1.add_to_collection('mktest_vd_long_term_kw_feed', input['mktest_vd_long_term_kw_feed'])
    input['mktest_vd_long_term_kw_feed'] += dummy

    input['mktest_long_search_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_long_search_kw_feed')
    tf.compat.v1.add_to_collection('mktest_long_search_kw_feed', input['mktest_long_search_kw_feed'])
    input['mktest_long_search_kw_feed'] += dummy

    input['mktest_new_search_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_new_search_kw_feed')
    tf.compat.v1.add_to_collection('mktest_new_search_kw_feed', input['mktest_new_search_kw_feed'])
    input['mktest_new_search_kw_feed'] += dummy

    input['mktest_user_kw_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'mktest_user_kw_feed')
    tf.compat.v1.add_to_collection('mktest_user_kw_feed', input['mktest_user_kw_feed'])
    input['mktest_user_kw_feed'] += dummy

    # mkyuwen add here
    input['mktest_distribution_id_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'mktest_distribution_id_feed')
    tf.compat.v1.add_to_collection('mktest_distribution_id_feed', input['mktest_distribution_id_feed'])
    input['mktest_distribution_id_feed'] += dummy
    input['mktest_distribution_id_feed'] = tf.squeeze(input['mktest_distribution_id_feed'], 1)


    # mkyuwen 0525
    dummy_float = tf.cast(dummy, tf.float32)
    input['mktest_new_search_kw_score_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.float32), [None, None], 'mktest_new_search_kw_score_feed')
    tf.compat.v1.add_to_collection('mktest_new_search_kw_score_feed', input['mktest_new_search_kw_score_feed'])
    input['mktest_new_search_kw_score_feed'] += dummy_float

    input['mktest_user_kw_score_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.float32), [None, None], 'mktest_user_kw_score_feed')
    tf.compat.v1.add_to_collection('mktest_user_kw_score_feed', input['mktest_user_kw_score_feed'])
    input['mktest_user_kw_score_feed'] += dummy_float

    input['mktest_doc_kw_score_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.float32), [None, None], 'mktest_doc_kw_score_feed')
    tf.compat.v1.add_to_collection('mktest_doc_kw_score_feed', input['mktest_doc_kw_score_feed'])
    input['mktest_doc_kw_score_feed'] += dummy_float

    input['mktest_doc_kw_secondary_score_feed'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.float32), [None, None], 'mktest_doc_kw_secondary_score_feed')
    tf.compat.v1.add_to_collection('mktest_doc_kw_secondary_score_feed', input['mktest_doc_kw_secondary_score_feed'])
    input['mktest_doc_kw_secondary_score_feed'] += dummy_float

    def _add(name, dtype=tf.int64, dims=None):
      if dims is None:
        dims = [None, 1]
      input[name] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=dtype), dims, '%s_feed' % name)
      tf.compat.v1.add_to_collection('%s_feed' % name, input[name])
      input[name] += dummy
      if dims[-1] == 1:
        input[name] = tf.squeeze(input[name], 1)
    
    def _adds(names, dtype=tf.int64, dims=None):
      for name in names:
        _add(name, dtype, dims)

    names_for_history = ['tw_history', 'tw_history_topic', 'tw_history_rec', 'tw_history_kw', 'vd_history', 'vd_history_topic']
    _adds(names_for_history, tf.int64, [None, None])

