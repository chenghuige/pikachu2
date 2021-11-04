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

from projects.ai.mango.src.config import *
from projects.ai.mango.src import util

class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__() 
    
    def _emb(vocab_name, emb_name=None):
      return util.create_emb(vocab_name, emb_name)

    self.uemb = _emb('did')

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    # TODO 还需要后期再确认用_emb 还是 util.create_image_emb()
    self.image_emb = _emb('image')

    self.time_emb = keras.layers.Embedding(1200, FLAGS.emb_size, name='time_emb')

    self.active_emb = keras.layers.Embedding(100, FLAGS.emb_size, name='active_emb')

    self.tlen_emb = keras.layers.Embedding(100, FLAGS.emb_size, name='tlen_emb')
    self.dur_emb = keras.layers.Embedding(100, FLAGS.emb_size, name='dur_emb')
    self.vv_emb = keras.layers.Embedding(100, FLAGS.emb_size, name='vv_emb')
    self.ctr_emb = keras.layers.Embedding(200, FLAGS.emb_size, name='ctr_emb')
    self.has_prev_emb = keras.layers.Embedding(5, FLAGS.emb_size, name='has_prev_emb')

    self.context_emb = melt.layers.QREmbedding(5000000, FLAGS.emb_size, num_buckets=500000, name='context_emb')
    self.item_emb = melt.layers.QREmbedding(5000000, FLAGS.emb_size, num_buckets=500000, name='item_emb')
    self.cross_emb = melt.layers.QREmbedding(FLAGS.cross_height, FLAGS.emb_size, num_buckets=FLAGS.num_buckets, name='cross_emb')

    self.position_emb = keras.layers.Embedding(100, FLAGS.emb_size, name='position_emb')

    self.sum_pooling = melt.layers.SumPooling()
    self.mean_pooling = melt.layers.MeanPooling()
    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.feat_pooling = melt.layers.Pooling(FLAGS.feat_pooling)

    self.dense = keras.layers.Dense(1)
    self.batch_norm = keras.layers.BatchNormalization()
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    mlp_dims = [FLAGS.emb_size * 2, FLAGS.emb_size] if not FLAGS.big_mlp else [FLAGS.emb_size * 4, FLAGS.emb_size * 2, FLAGS.emb_size]
    self.dense_mlp = melt.layers.MLP(mlp_dims,
                                     activation=activation, 
                                     drop_rate=FLAGS.mlp_dropout,
                                     name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,
                                     drop_rate=FLAGS.mlp_dropout,
                                     name='image_mlp')
    mlp_dims = [512, 256, 64] if not FLAGS.big_mlp else [1024, 512, 256]
    self.mlp = melt.layers.MLP(mlp_dims, activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.title_encoder = util.get_encoder(FLAGS.title_encoder)
    self.story_encoder = util.get_encoder(FLAGS.story_encoder)
    self.his_encoder = util.get_encoder(FLAGS.his_encoder)
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)
    self.his_pooling = util.get_att_pooling(FLAGS.his_pooling)
    self.his_pooling2 = util.get_att_pooling(FLAGS.his_pooling2)
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)
    self.stars_encoder = util.get_encoder(FLAGS.stars_encoder)

    # self.stars_pooling = util.get_att_pooling(FLAGS.stars_pooling)
    self.stars_pooling = melt.layers.Pooling(FLAGS.stars_pooling)
    self.stars_att_pooling = util.get_att_pooling(FLAGS.stars_att_pooling)
    self.title_pooling = util.get_att_pooling(FLAGS.title_pooling) if FLAGS.title_att else melt.layers.Pooling(FLAGS.title_pooling)
    self.image_encoder = util.get_encoder(FLAGS.image_encoder)
    self.image_pooling = util.get_att_pooling(FLAGS.image_pooling)

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
    wv_len = tf.cast(melt.length(input['watch_vids']), tf.float32)
    wv_len2 = tf.cast(melt.length(input['cids']), tf.float32)
    
    if FLAGS.use_dense_common:
      # 文章侧特征
      ctr_ = melt.scalar_feature(input['ctr'] )
      vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
      vdur = input['duration']
      vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
      title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
      twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
      num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
      fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
      fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)

      cid_rate_ = melt.scalar_feature(input['cid_rate'])
    
      feats += [
        ctr_, vv_, vdur_, title_len_, twords_len_,
        num_stars_, fresh_, cid_rate_
      ]

    if FLAGS.use_dense_history:
      # 用户侧
      # 用户阅读历史个数 表示一定用户活跃度 比较重要
      num_hists_ = melt.scalar_feature(wv_len + 1, max_val=51, scale=True)
      num_hists2_ = melt.scalar_feature(wv_len2 + 1, max_val=51, scale=True)
      num_shows_ = melt.scalar_feature(input['num_shows'], max_val=20000, scale=True)

      feats += [num_hists_, num_hists2_, num_shows_]

    #   # wv_len2 = input['hits']
    if FLAGS.use_dense_his_durs:
      # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
      vdur = input['duration']
      durs = input['durations']
      avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (wv_len2 + 1)
      avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

      delta_durs = tf.math.abs(vdur - avg_durs)
      delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

      # # 用户最近点击视频的dur
      last_dur = input['durations'][:,0]
      last_dur_ =  melt.scalar_feature(last_dur, max_val=10000, scale=True)

      last_delta_dur = tf.math.abs(vdur - last_dur)
      last_delta_dur_ = melt.scalar_feature(last_delta_dur, max_val=10000, scale=True)

      feats += [
        avg_durs_, delta_durs_,
        last_dur_, last_delta_dur_
      ]
    
    if FLAGS.use_dense_his_freshes:
      # 待最后验证7天simple模型看有一点点下降 没收益
      # if FLAGS.use_his_freshes:
      # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
      fresh = tf.cast(input['fresh'], tf.float32)
      freshes = tf.cast(tf.math.abs(input['freshes']), tf.float32) / (3600 * 24.)
      avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (wv_len2 + 1)
      avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
      delta_freshes = tf.math.abs(fresh - avg_freshes)
      delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

      # 用户最近点击的fresh
      last_fresh = freshes[:,0]
      last_fresh_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
      last_delta_fresh = tf.math.abs(fresh - last_fresh)
      last_delta_fresh_ = melt.scalar_feature(last_delta_fresh, max_val=1200, scale=True)

      feats += [
        avg_freshes_, delta_freshes_, last_delta_fresh_, 
      ]
    
    if FLAGS.use_dense_his_interval:
      # 用户最近一刷距今时间 天级别
      last_interval = tf.cast(tf.cast((input['timestamp'] - input['watch_times'][:,0]) / 3600, tf.int32), tf.float32)
      last_interval_ = melt.scalar_feature(last_interval, max_val=1200, scale=True)

      old_interval = tf.cast(tf.cast((input['timestamp'] - input['watch_times'][:,-1]) / 3600, tf.int32), tf.float32)
      old_interval_ = melt.scalar_feature(old_interval, max_val=2400, scale=True)

      feats += [
        last_interval_, old_interval_
      ]

    # # 用户点击时间delta序列
    timestamp = melt.tile_by(input['timestamp'], input['watch_times'])

    if FLAGS.use_dense_prev_info:
      # cur_time = tf.expand_dims(input['timestamp'], 1)
      # his_times = tf.concat([cur_time, input['watch_times']], 1)
      # his_intervals = tf.cast(his_times[:,:-1] - his_times[:,1:], tf.float32)
      # his_intervals_l1 = his_intervals / tf.reduce_sum(his_intervals, -1, keepdims=True)
      # his_intervals_l2 = tf.nn.l2_normalize(his_intervals, 1)

      prev = melt.tile_by(input['prev'], input['watch_vids'])
      mask = tf.cast(tf.not_equal(prev, 0), tf.float32)
      prev_mach = tf.reduce_sum(tf.cast(tf.equal(prev, input['watch_vids']), tf.float32) * mask, 1)
      prev_match_ = melt.scalar_feature(prev_mach, max_val=51, scale=True)

      prev_ctr_ = melt.scalar_feature(input['prev_ctr'])
      prev_vv_ = melt.scalar_feature(input['prev_vv'], max_val=100000, scale=True)
      prev_dur_ = melt.scalar_feature(input['prev_duration_'], max_val=10000, scale=True)
      prev_title_len_ = melt.scalar_feature(input['prev_title_length_'], max_val=205, scale=True)

      feats += [
        prev_match_, prev_ctr_, prev_dur_, prev_title_len_
      ]

    his_intervals = tf.cast(tf.cast((timestamp - input['watch_times']) / (3600 * 24), tf.int32), tf.float32)

    if FLAGS.use_dense_his_clicks:
      ## TODO 这里问题是长度不确定 tfrecord生成的时候都弄成50 ？ 或者生成tfrecord的时候做这个特征也行
      # normed_his_intervals = tf.math.minimum(his_intervals / 365., 1.)
      # # 用户一天内的点击次数
      # last_1day_clicks = tf.reduce_sum(tf.cast(his_intervals <= 1, tf.float32), 1)
      # last_1day_clicks_ = melt.scalar_feature(last_1day_clicks, max_val=50, scale=True)
      # 用户3天之内点击次数
      last_3day_clicks = tf.reduce_sum(tf.cast(his_intervals <= 3, tf.float32), 1)
      last_3day_clicks_ = melt.scalar_feature(last_3day_clicks, max_val=50, scale=True)
      # 用户一周之内点击次数 
      last_7day_clicks = tf.reduce_sum(tf.cast(his_intervals <= 7, tf.float32), 1)
      last_7day_clicks_ = melt.scalar_feature(last_7day_clicks, max_val=50, scale=True)
      # 用户一个月之内点击次数
      last_30day_clicks = tf.reduce_sum(tf.cast(his_intervals <= 30, tf.float32), 1)
      last_30day_clicks_ = melt.scalar_feature(last_30day_clicks, max_val=50, scale=True)

      feats += [
        last_3day_clicks_, last_7day_clicks_, last_30day_clicks_
        ]

    if FLAGS.use_dense_match:
      cur_first_stars = melt.tile_by(input['stars'][:,0], input['watch_vids'])
      # print(cur_first_stars)
      match_first_stars = tf.cast(tf.reduce_sum(input['first_stars_list'] - cur_first_stars, 1), tf.float32) / (wv_len2 + 1)
      match_first_stars_ = melt.scalar_feature(match_first_stars, max_val=51, scale=True)

      cur_cid = melt.tile_by(input['cid'], input['watch_vids'])
      match_cid = tf.cast(tf.reduce_sum(input['cids'] - cur_cid, 1), tf.float32)/ (wv_len2 + 1)
      match_cid_ = melt.scalar_feature(match_cid, max_val=51, scale=True)

      cur_class_id = melt.tile_by(input['class_id'], input['watch_vids'])
      match_class_id = tf.cast(tf.reduce_sum(input['class_ids'] - cur_class_id, 1), tf.float32)/ (wv_len2 + 1)
      match_class_id_ = melt.scalar_feature(match_class_id, max_val=51, scale=True)

      cur_second_class = melt.tile_by(input['second_class'], input['watch_vids'])
      match_second_class = tf.cast(tf.reduce_sum(input['second_classes'] - cur_second_class, 1), tf.float32)/ (wv_len2 + 1)
      match_second_class_ = melt.scalar_feature(match_second_class, max_val=51, scale=True)

      cur_is_intact = melt.tile_by(input['is_intact'], input['watch_vids'])
      match_is_intact = tf.cast(tf.reduce_sum(input['is_intacts'] - cur_is_intact, 1), tf.float32)/ (wv_len2 + 1)
      match_is_intact_ = melt.scalar_feature(match_is_intact, max_val=51, scale=True)

      feats += [
        match_first_stars_, 
        match_cid_,
        match_class_id_,
        match_second_class_,
        match_is_intact_, 
      ]

    # print('dense_feats-----------', len(feats))
    feats = tf.concat(feats, -1)
    dense_emb = self.dense_mlp(feats)
    return dense_emb


  def call(self, input):
    embs = []

    wv_len = melt.length(input['watch_vids'])
    wv_len = tf.math.maximum(wv_len, 1)
    # user 
    if FLAGS.use_uid:
      uemb = self.uemb(input['did'])
      # uemb = self.dense_uemb(uemb)

      embs += [uemb]

    if FLAGS.use_vid:
      # video
      vemb = self.vemb(input['vid'])
      embs += [vemb]

    if FLAGS.use_last_vid:
      last_emb = self.vemb(input['watch_vids'][:,0])
      embs += [last_emb]

    if FLAGS.use_uinfo:
      # user info
      remb = self.remb(input['region'])
      #   phone
      pmod_emb = self.pmod_emb(input['mod'])
      pmf_emb = self.pmf_emb(input['mf'])
      psver_emb = self.psver_emb(input['sver'])
      paver_emb = self.paver_emb(input['aver'])

      embs += [remb, pmod_emb, pmf_emb, psver_emb, paver_emb]
  
    # wvemb = self.pooling(self.vid_encoder(wvembs, wv_len), wv_len)
    if FLAGS.use_prev_info:
      prev_emb = self.vemb(input['prev'])
      prev_intact_emb = self.intact_emb(input['prev_is_intact'])

      embs += [prev_emb, prev_intact_emb]

    if FLAGS.use_class_info:
      cemb = self.cemb(input['cid'])
      class_emb = self.class_emb(input['class_id'])
      second_class_emb = self.second_class_emb(input['second_class'])
      intact_emb = self.intact_emb(input['is_intact'])

      embs += [cemb, class_emb, second_class_emb, intact_emb]

    if FLAGS.use_history_class:
      # 最近点击的类别 simple模型验证有效
      cembs = self.cemb(input['cids'])
      class_embs = self.class_emb(input['class_ids'])
      second_class_embs = self.second_class_emb(input['second_classes'])
      intact_embs = self.intact_emb(input['is_intacts'])

      his_cemb = self.pooling(cembs)
      his_class_emb = self.pooling(class_embs)
      his_second_class_emb = self.pooling(second_class_embs)
      his_intact_emb = self.pooling(intact_embs)

      embs += [his_cemb, his_class_emb, his_second_class_emb, his_intact_emb]

      if FLAGS.use_last_class:
        last_cemb = self.cemb(input['cids'][:,0])
        last_class_emb = self.class_emb(input['class_ids'][:,0])
        last_second_class_emb = self.second_class_emb(input['second_classes'][:,0])
        last_intact_emb = self.intact_emb(input['is_intacts'][:,0])
        embs += [last_cemb, last_class_emb, last_second_class_emb, last_intact_emb]

    if FLAGS.use_stars:
      # video info
      stars_embs = self.stars_emb(input['stars'])
      stars_embs = self.stars_encoder(stars_embs)
      # if not FLAGS.stars_att:
      stars_emb = self.stars_pooling(stars_embs, melt.length(input['stars']))
      # else:
      #   stars_emb = self.stars_pooling(his_emb, stars_embs, melt.length(input['stars']))

      embs += [stars_emb]

      if FLAGS.use_all_stars:
        all_stars_embs = self.stars_emb(input['all_stars_list'])
        all_stars_embs = self.stars_encoder(all_stars_embs)
        all_stars_emb = self.stars_pooling(all_stars_embs, melt.length(input['all_stars_list']))
        # all_stars_emb = self.stars_att_pooling(stars_emb, all_stars_embs, melt.length(input['all_stars_list']))

        embs += [all_stars_emb]

      if FLAGS.use_first_star:
        first_star_emb = stars_embs[:,0]
        embs += [first_star_emb]

      if FLAGS.use_stars_list:
        first_stars_embs = self.stars_emb(input['first_stars_list'])
        first_stars_embs = self.stars_encoder(first_stars_embs)
        first_stars_emb = self.stars_pooling(first_stars_embs)

        embs += [first_stars_emb]

      if FLAGS.use_last_stars:
        last_stars_embs = self.stars_emb(input['last_stars'])
        last_stars_embs = self.stars_encoder(last_stars_embs)
        if not FLAGS.stars_att:
          last_stars_emb = self.stars_pooling(last_stars_embs, melt.length(input['last_stars']))
        else:
          last_stars_emb = self.stars_att_pooling(his_emb, last_stars_embs, melt.length(input['last_stars']))
        
        last_first_star_emb = last_stars_embs[:,0]

        embs += [last_stars_emb, last_first_star_emb]

    if FLAGS.use_title:
      title_embs = self.words_emb(input['title'])
      title_embs = self.title_encoder(title_embs)
      if not FLAGS.title_att:
        title_emb = self.title_pooling(title_embs, melt.length(input['title']))
      else:
        title_emb = self.title_pooling(his_emb, title_embs, melt.length(input['title']))

      embs += [title_emb]

      if FLAGS.use_titles:
        titles = input['titles']
        titles_len = melt.length(titles)
        titles_embs = self.words_emb(titles)
        titles_emb = self.title_pooling(titles_embs, titles_len)
        
        embs += [titles_emb]

      if FLAGS.use_last_title:
        last_title = input['last_title']
        last_title_len = melt.length(last_title)
        last_title_embs = self.words_emb(last_title)
        last_title_emb = self.title_pooling(last_title_embs, last_title_len)

        embs += [last_title_emb]

    if FLAGS.use_story:
      story = input['story']
      story_len = melt.length(story)
      story_embs = self.words_emb(story)
      story_embs = self.story_encoder(story_embs)
      story_emb = self.title_pooling(story_embs, story_len)

      embs += [story_emb]

    if FLAGS.use_image:
      # image_emb = input['image_emb']
      # image_emb = tf.reshape(image_emb, [-1, 128])
      image_emb = self.image_emb(input['vid'])
      image_emb = self.image_mlp(image_emb)

      embs += [image_emb]

      if FLAGS.use_his_image:
        his_image_embs = self.image_emb(input['watch_vids'])
        his_image_embs = self.image_mlp(his_image_embs)
        his_image_embs = self.image_encoder(his_image_embs)
        his_image_emb = self.pooling(his_image_embs, wv_len)

        embs += [his_image_emb]

    if FLAGS.use_active:      
      active_emb = self.active_emb(wv_len)

      embs += [active_emb]

    if FLAGS.use_others:
      tlen_emb = self.tlen_emb(input['title_length_'])
      dur_emb = self.dur_emb(input['duration_'])
      # print('vv', input['vv_'])
      vv_emb = self.vv_emb(input['vv_'])
      ctr_emb = self.ctr_emb(input['ctr_'])

      embs += [tlen_emb, dur_emb, vv_emb, ctr_emb]

      if FLAGS.use_prev_info:
        prev_tlen_emb = self.tlen_emb(input['prev_title_length_'])
        prev_dur_emb = self.dur_emb(input['prev_duration_'])
        # print('vv', input['vv_'])
        prev_vv_emb = self.vv_emb(input['prev_vv_'])
        prev_ctr_emb = self.ctr_emb(input['prev_ctr_'])

        has_prev_emb = self.has_prev_emb(input['has_prev'])

        embs += [prev_tlen_emb, prev_dur_emb, prev_ctr_emb, has_prev_emb]

    if FLAGS.use_dense:
      dense_emb = self.deal_dense(input)

      embs += [dense_emb]

    if FLAGS.use_history:
      # print(wv_len)
      watch_vids = util.unk_aug(input['watch_vids'])
      #     0) Invalid argument: Incompatible shapes: [512,50,128] vs. [512,49,128]
      #  [[node model/add_1 (defined at /tmp/tmpg_vuyuou.py:546) ]]
      #  [[sigmoid_cross_entropy_loss/value/_431]]
      # FIXME now just hack
      if FLAGS.his_strategy == 'bst' or FLAGS.his_pooling == 'mhead':
        # watch_vids = watch_vids[:,:tf.reduce_max(wv_len)]
        mask = tf.cast(tf.equal(watch_vids, 0), tf.int64)
        watch_vids += mask
        wv_len = tf.ones_like(wv_len) * 50
      wvembs = self.vemb(watch_vids)

      timestamp = tf.tile(tf.expand_dims(input['timestamp'],1), [1, tf.shape(input['watch_times'])[1]])
      his_intervals = tf.cast(tf.math.maximum((timestamp - input['watch_times']), 0) / (3600 * 24), tf.int32)
      # his_intervals = tf.cast(tf.math.maximum((timestamp - input['watch_times']), 0) / 3600, tf.int32)
      # his_intervals = tf.cast((timestamp / (3600 * 24), tf.int32) - tf.cast((input['watch_times'] / (3600 * 24), tf.int32)
      his_intervals = tf.math.minimum(his_intervals, 1199)
      his_interval_embs = self.time_emb(his_intervals)
      now_time_emb = self.time_emb(tf.zeros_like(input['vid']))      

      his_embs = wvembs
      his_embs = self.his_encoder(his_embs, wv_len)
      
      if FLAGS.use_position_emb:
        bs = melt.get_shape(his_embs, 0)
        postions = tf.tile(tf.expand_dims(tf.range(50), 0),[bs, 1])
        position_embs = self.position_emb(postions)
        his_embs += position_embs

      if FLAGS.use_time_emb:
        his_embs += his_interval_embs

      if FLAGS.his_strategy == '1': # simple
        if FLAGS.his_encoder:
          # cur_emb = self.his_encoder(tf.expand_dims(cur_emb, 1), tf.ones_like(wv_len))
          cur_emb = self.his_encoder(tf.stack([vemb, prev_emb], 1), tf.ones_like(wv_len) * 2)
          # cur_emb = tf.squeeze(cur_emb, 1)
          vemb, prev_emb = tf.split(cur_emb, 2, axis=1)
          vemb = tf.squeeze(vemb, 1)
          prev_emb = tf.squeeze(prev_emb, 1)
      elif FLAGS.his_strategy == '2': # merge other info
        his_embs = tf.concat([wvembs, cembs], axis=-1)
        his_embs = self.his_encoder(his_embs, wv_len)
        cur_emb = tf.concat([vemb, cemb], axis=-1)
        if FLAGS.his_encoder:
          cur_emb = self.his_encoder(tf.expand_dims(cur_emb, 1), tf.ones_like(wv_len))
        else:
          his_embs = self.his_dense(his_embs)
      elif FLAGS.his_strategy == '3': # merge other info
        his_embs = tf.concat([wvembs, cembs, class_embs, second_class_embs, intact_embs], axis=-1)
        his_embs = self.his_dense(self.his_encoder(his_embs, wv_len))
        cur_emb = tf.concat([vemb, cemb, class_emb, second_class_emb, intact_emb], axis=-1)
        if FLAGS.his_encoder:
          cur_emb = self.his_encoder(tf.expand_dims(cur_emb, 1), tf.ones_like(wv_len))
        else:
          his_embs = self.his_dense(his_embs)
      elif FLAGS.his_strategy == '4': # fusion other info
        his_embs = self.fusion(wvembs, [titles_embs])
        his_embs = self.his_dense(self.his_encoder(his_embs, wv_len))
        cur_emb = self.cur_dense(self.fusion(vemb, [title_emb]))
      elif FLAGS.his_strategy.startswith('bst'): # transformer
        his_embs = wvembs
        his_embs = tf.concat([tf.expand_dims(vemb, 1), tf.expand_dims(prev_emb, 1), his_embs], 1)
        # time_embs = tf.concat([tf.expand_dims(now_time_emb, 1), tf.expand_dims(now_time_emb, 1), his_interval_embs], 1)
        # print(his_embs, time_embs)
        postions = tf.tile(tf.expand_dims(tf.range(52), 0),[bs, 1])
        position_embs = self.position_emb(postions)
        his_embs += position_embs
        # his_embs = self.transformer(his_embs, wv_len + 2, time_embs)
        his_embs = self.transformer(his_embs, wv_len + 2)
        # his_embs = self.transformer(his_embs, 52, time_embs)

        vemb = his_embs[:,0,:]
        prev_emb = his_embs[:,1,:]
        his_embs = his_embs[:,2:,:]

      self.his_embs = his_embs
      his_concat = tf.reshape(his_embs, [-1, 50 * melt.get_shape(his_embs, -1)])
    
      his_emb = self.his_pooling(vemb, his_embs, wv_len)
      
      embs += [his_emb]

    if FLAGS.use_contexts:
      context_embs = self.context_emb(input['context'])
      # num_contexts = int(len(context_cols) * (len(context_cols) - 1) / 2)
      # context_embs = tf.unstack(context_embs, axis=1)
      # embs += context_embs
      context_emb = self.pooling(context_embs)

      embs += [context_emb]

    if FLAGS.use_items:
      item_embs = self.item_emb(input['item'])
      item_emb = self.pooling(item_embs)
      embs += [item_emb]

    if FLAGS.use_crosses:
      cross_embs = self.cross_emb(input['cross'])
      cross_emb = self.pooling(cross_embs)
      embs += [cross_emb]

    if FLAGS.use_shows:
      show_embs = self.vemb(input['show_vids'])
      if not FLAGS.his_pooling2:
        show_emb = self.pooling(show_embs, melt.length(input['show_vids']))
      else:
        show_emb = self.his_pooling2(vemb, show_embs, melt.length(input['show_vids']))
      embs += [show_emb]

    # print('-----------embs:', len(embs))
    embs = tf.stack(embs, axis=1)

    if FLAGS.batch_norm:
      embs = self.batch_norm(embs)

    if FLAGS.l2_normalize_before_pooling:
      x = tf.math.l2_normalize(embs)

    x = self.feat_pooling(embs)

    if FLAGS.use_dense:
      x = tf.concat([x, dense_emb], axis=1)

    if FLAGS.use_his_concat:
      x = tf.concat([x, his_concat], axis=1)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    self.watches = melt.length(input['watch_vids'])
    return self.logit
