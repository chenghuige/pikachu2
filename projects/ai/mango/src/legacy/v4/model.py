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
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    if FLAGS.use_uid:
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

    self.tlen_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='tlen_emb')
    self.dur_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='dur_emb')
    self.vv_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='vv_emb')
    self.ctr_emb = keras.layers.Embedding(100, FLAGS.emb_size, name='ctr_emb')
    self.has_prev_emb = keras.layers.Embedding(2, FLAGS.emb_size, name='has_prev_emb')

    if FLAGS.use_contexts:
      self.context_emb = melt.layers.QREmbedding(5000000, FLAGS.emb_size, num_buckets=500000, name='context_emb')
      self.item_emb = melt.layers.QREmbedding(5000000, FLAGS.emb_size, num_buckets=500000, name='item_emb')
      self.cross_emb = melt.layers.QREmbedding(5000000, FLAGS.emb_size, num_buckets=FLAGS.num_buckets, name='cross_emb')

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

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
    self.his_encoder = util.get_encoder(FLAGS.his_encoder)
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)
    self.his_pooling = util.get_att_pooling(FLAGS.his_pooling)
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

    self.stars_pooling = util.get_att_pooling(FLAGS.stars_pooling)
    self.title_pooling = util.get_att_pooling(FLAGS.title_pooling) if FLAGS.title_att else melt.layers.Pooling(FLAGS.title_pooling)
    self.image_encoder = util.get_encoder(FLAGS.image_encoder)
    self.image_pooling = util.get_att_pooling(FLAGS.image_pooling)

    if FLAGS.his_strategy.startswith('bst'):
      self.transformer = melt.layers.transformer.Encoder(num_layers=1, d_model=FLAGS.hidden_size, num_heads=FLAGS.num_heads, 
                                                         dff=FLAGS.hidden_size, maximum_position_encoding=None, activation=FLAGS.transformer_activation,
                                                         rate=FLAGS.transformer_dropout)

    self.fusion = melt.layers.SemanticFusion(drop_rate=0.1)

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

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
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
    wv_len = tf.cast(melt.length(input['watch_vids']), tf.float32)
    # 用户侧
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(wv_len, max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.abs(vdur - avg_durs)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户最近点击视频的dur
    last_dur = input['durations'][:,0]
    last_dur_ =  melt.scalar_feature(last_dur, max_val=10000, scale=True)

    last_delta_dur = tf.math.abs(vdur - last_dur)
    last_delta_dur_ = melt.scalar_feature(last_delta_dur, max_val=10000, scale=True)
    
    # 待最后验证7天simple模型看有一点点下降 没收益
    if FLAGS.use_his_freshes:
      # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
      freshes = tf.math.abs(input['freshes']) / (3600 * 24)
      avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
      avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
      delta_freshes = tf.math.abs(fresh - avg_freshes)
      delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

      # 用户最近点击的fresh
      last_fresh = freshes[:,0]
      last_fresh_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
      last_delta_fresh = tf.math.abs(fresh - last_fresh)
      last_delta_fresh_ = melt.scalar_feature(last_delta_fresh, max_val=1200, scale=True)

    # 用户最近一刷距今时间 天级别
    last_interval = tf.cast(tf.cast((input['timestamp'] - input['watch_times'][:,0]) / 3600, tf.int32), tf.float32)
    last_interval_ = melt.scalar_feature(last_interval, max_val=1200, scale=True)

    old_interval = tf.cast(tf.cast((input['timestamp'] - input['watch_times'][:,-1]) / 3600, tf.int32), tf.float32)
    old_interval_ = melt.scalar_feature(old_interval, max_val=2400, scale=True)

    # 用户点击时间delta序列
    timestamp = tf.tile(tf.expand_dims(input['timestamp'],1), [1, tf.shape(input['watch_times'])[1]])
    his_intervals = tf.cast(tf.cast((timestamp - input['watch_times']) / (3600 * 24), tf.int32), tf.float32)
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

    if FLAGS.use_match:
      match_stars_ = melt.scalar_feature(input['match_stars'], max_val=50, scale=True)
      match_stars_rate_ = melt.scalar_feature(tf.cast(input['match_stars'], tf.float32)/ (wv_len + 0.0001))
      match_cids_ = melt.scalar_feature(input['match_cids'], max_val=50, scale=True)
      match_cids_rate_ = melt.scalar_feature(tf.cast(input['match_cids'], tf.float32) / (wv_len + 0.0001))
      match_class_ids_ = melt.scalar_feature(input['match_class_ids'], max_val=50, scale=True)
      match_class_ids_rate_ = melt.scalar_feature(tf.cast(input['match_class_ids'], tf.float32) / (wv_len + 0.0001))
      match_second_classes_ = melt.scalar_feature(input['match_second_classes'], max_val=50, scale=True)
      match_second_classes_rate_ = melt.scalar_feature(tf.cast(input['match_second_classes'], tf.float32) / (wv_len + 0.0001))
      match_is_intacts_ = melt.scalar_feature(input['match_is_intacts'], max_val=50, scale=True)
      match_is_intacts_rate_ = melt.scalar_feature(tf.cast(input['match_is_intacts'], tf.float32) / (wv_len + 0.0001))
      match_prev_ = melt.scalar_feature(input['match_prev'], max_val=50, scale=True)
      match_prev_rate_ = melt.scalar_feature(tf.cast(input['match_prev'], tf.float32) / (wv_len + 0.0001))
      match_first_word_ = melt.scalar_feature(input['match_first_word'], max_val=50, scale=True)
      match_first_word_rate_ = melt.scalar_feature(tf.cast(input['match_first_word'], tf.float32) / (wv_len + 0.0001))

      match_last_stars_ = melt.scalar_feature(input['match_last_stars'], max_val=50, scale=True)
      match_last_stars_rate_ = melt.scalar_feature(tf.cast(input['match_last_stars'], tf.float32) / (wv_len + 0.0001))
      match_last_cids_ = melt.scalar_feature(input['match_last_cids'], max_val=50, scale=True)
      match_last_cids_rate_ = melt.scalar_feature(tf.cast(input['match_last_cids'], tf.float32) / (wv_len + 0.0001))
      match_last_class_ids_ = melt.scalar_feature(input['match_last_class_ids'], max_val=50, scale=True)
      match_last_class_ids_rate_ = melt.scalar_feature(tf.cast(input['match_last_class_ids'], tf.float32)/ (wv_len + 0.0001))
      match_last_second_classes_ = melt.scalar_feature(input['match_last_second_classes'], max_val=50, scale=True)
      match_last_second_classes_rate_ = melt.scalar_feature(tf.cast(input['match_last_second_classes'], tf.float32)/ (wv_len + 0.0001))
      match_last_is_intacts_ = melt.scalar_feature(input['match_last_is_intacts'], max_val=50, scale=True)
      match_last_is_intacts_rate_ = melt.scalar_feature(tf.cast(input['match_last_is_intacts'], tf.float32) / (wv_len + 0.0001))
      match_last_prev_ = melt.scalar_feature(input['match_last_prev'], max_val=50, scale=True)
      match_last_prev_rate_ = melt.scalar_feature(tf.cast(input['match_last_prev'], tf.float32) / (wv_len + 0.0001))
      match_last_first_word_ = melt.scalar_feature(input['match_last_first_word'], max_val=50, scale=True)
      match_last_first_word_rate_ = melt.scalar_feature(tf.cast(input['match_last_first_word'], tf.float32) / (wv_len + 0.0001))

    dense_feats = [
                    ctr_, vv_, vdur_, title_len_, twords_len_, 
                    num_stars_, fresh_, num_hists_, 
                    avg_durs_, delta_durs_,
                    last_dur_, last_delta_dur_, 
                    last_interval_, old_interval_,
                    last_3day_clicks_, last_7day_clicks_, last_30day_clicks_
                  ]
    if FLAGS.use_his_freshes:
      dense_feats += [
                        avg_freshes_, delta_freshes_, 
                        last_fresh_, delta_last_fresh
                     ]
    if FLAGS.use_match:
      dense_feats += [
        match_stars_, match_stars_rate_, 
        match_cids_, match_cids_rate_, 
        match_class_ids_, match_class_ids_rate_,
        match_second_classes_, match_second_classes_rate_,
        match_is_intacts_, match_is_intacts_rate_,
        match_prev_, match_prev_rate_,
        match_first_word_, match_first_word_rate_,
      ]
    dense_feats = tf.concat(dense_feats, -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    if FLAGS.use_uid:
      uemb = self.uemb(input['did'])
      # uemb = self.dense_uemb(uemb)
  
    wv_len = melt.length(input['watch_vids'])
    # print(wv_len)
    watch_vids = util.unk_aug(input['watch_vids'])
    wvembs = self.vemb(watch_vids)
    
    # wvemb = self.sum_pooling(self.vid_encoder(wvembs, wv_len), wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    # video
    vemb = self.vemb(input['vid'])
    last_vemb = self.vemb(input['watch_vids'][:,0])
    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # 最近点击的类别 simple模型验证有效
    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])

    last_cemb = self.cemb(input['cids'][:,0])
    last_class_emb = self.class_emb(input['class_ids'][:,0])
    last_second_class_emb = self.second_class_emb(input['second_classes'][:,0])
    last_intact_emb = self.intact_emb(input['is_intacts'][:,0])
    last_vcemb = last_cemb + last_class_emb + last_second_class_emb + last_intact_emb
   
    if FLAGS.use_titles:
      titles = input['titles']
      titles_len = melt.length(titles)
      titles_len = tf.math.maximum(titles_len, 50)
      titles_embs = self.words_emb(titles[:,:50])
      # batch_size = melt.get_shape(input['vid'], 0)
      # titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
      # titles_embs = self.sum_pooling(titles_embs)
      # titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])
      # titles_embs = self.title_encoder(titles_embs, wv_len)
      # titles_emb = self.sum_pooling(titles_embs, wv_len)
      # titles_embs = self.title_encoder(titles_embs, titles_len)
      # titles_emb = self.title_pooling(titles_embs, titles_len)
      titles_emb = self.sum_pooling(titles_embs, wv_len)

    story = input['story']
    story_len = melt.length(story)
    # story_len = tf.math.maximum(story_len, 50)
    story_embs = self.words_emb(story)
    # story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
    # story_embs = self.title_encoder(story_embs, story_len)
    # story_emb = self.title_pooling(story_embs, story_len)
    story_emb = self.sum_pooling(story_embs, story_len)

    if FLAGS.use_image:
      # image_emb = input['image_emb']
      # image_emb = tf.reshape(image_emb, [-1, 128])
      image_emb = self.image_emb(input['vid'])
      image_emb = self.image_mlp(image_emb)

      if FLAGS.use_his_image:
        his_image_embs = self.image_emb(input['watch_vids'])
        his_image_embs = self.image_mlp(his_image_embs)
        his_image_embs = self.image_encoder(his_image_embs)
        # his_image_emb = melt.layers.MeanPooling()(his_image_embs, tf.cast(wv_len, tf.float32) + 0.0001)
        his_image_emb = self.sum_pooling(his_image_embs, wv_len)

    timestamp = tf.tile(tf.expand_dims(input['timestamp'],1), [1, tf.shape(input['watch_times'])[1]])
    # his_intervals = tf.cast(tf.math.maximum((timestamp - input['watch_times']), 0) / (3600 * 24), tf.int32)
    his_intervals = tf.cast(tf.math.maximum((timestamp - input['watch_times']), 0) / 3600, tf.int32)
    # his_intervals = tf.cast((timestamp / (3600 * 24), tf.int32) - tf.cast((input['watch_times'] / (3600 * 24), tf.int32)
    his_intervals = tf.math.minimum(his_intervals, 1199)
    his_interval_embs = self.time_emb(his_intervals)
    now_time_emb = self.time_emb(tf.zeros_like(input['vid']))

    if FLAGS.his_strategy == '1': # simple
      his_embs = wvembs
      his_embs = self.his_encoder(his_embs, wv_len)
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
      time_embs = tf.concat([tf.expand_dims(now_time_emb, 1), tf.expand_dims(now_time_emb, 1), his_interval_embs], 1)
      his_embs = self.transformer(his_embs, wv_len + 2, time_embs)

      vemb = his_embs[:,0,:]
      prev_emb = his_embs[:,1,:]
      his_embs = his_embs[:,2:,:]

    self.his_embs = his_embs
    
    if FLAGS.his_strategy != 'bst2':
      his_emb = self.his_pooling(vemb, his_embs, wv_len)
      # his_emb2 = self.his_pooling(prev_emb, his_emb, wv_len)
    else:
      his_emb = self.sum_pooling(his_embs, wv_len)
      # his_emb2 = melt.layers.ConcatPooling()(his_embs)

    # video info
    stars_embs = self.stars_emb(input['stars'])
    if not FLAGS.stars_att:
      stars_emb = self.sum_pooling(stars_embs, melt.length(input['stars']))
    else:
      stars_emb = self.stars_pooling(his_emb, stars_embs, melt.length(input['stars']))

    # FLAGS.use_latest_stars = True
    if FLAGS.use_latest_stars:
      latest_stars_emb = self.stars_emb(input['stars_list'][:,0])

    # 这些由于长度最大500 所以会比较慢。。 缩短？
    if FLAGS.use_stars_list:
      stars_list_embs = self.stars_emb(input['stars_list'])
      stars_list_emb = self.stars_pooling(stars_emb, stars_list_embs, meltl.length(input['stars_list']))

    title_embs = self.words_emb(input['title'])
    title_embs = self.title_encoder(title_embs)
    if not FLAGS.title_att:
      if FLAGS.title_pooling != 'att':
        title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
      else:
        title_emb = self.title_pooling(title_embs, melt.length(input['title']))
    else:
      title_emb = self.title_pooling(his_emb, title_embs, melt.length(input['title']))

    # print('---', wv_len)
    active_emb = self.active_emb(wv_len)

    tlen_emb = self.tlen_emb(input['title_length_'])
    dur_emb = self.dur_emb(input['duration_'])
    # print('vv', input['vv_'])
    vv_emb = self.vv_emb(input['vv_'])
    ctr_emb = self.ctr_emb(input['ctr_'])
    has_prev_emb = self.has_prev_emb(input['has_prev'])

    dense_emb = self.deal_dense(input)

    embs = [
            remb, pmod_emb, pmf_emb, psver_emb, paver_emb, pemb, 
            vemb, last_vemb, prev_emb, 
            cemb, class_emb, second_class_emb, intact_emb, vcemb,
            last_cemb, last_class_emb, last_second_class_emb, last_intact_emb, last_vcemb,
            stars_emb, 
            title_emb, 
            story_emb, 
            his_emb, 
            # his_emb2,
            active_emb,
            tlen_emb, vv_emb, 
            dur_emb, ctr_emb,
            has_prev_emb,
            dense_emb
            ]
    
    if FLAGS.use_uid:
      embs += [uemb]

    if FLAGS.use_image:
      embs += [image_emb]
      if FLAGS.use_his_image:
        embs += [his_image_emb]

    if FLAGS.use_titles:
      embs += [titles_emb]

    if FLAGS.use_stars_list:
      embs += [stars_list_emb]

    if FLAGS.use_latest_stars:
      embs += [latest_stars_emb]

    if FLAGS.uv_only:
      embs = [his_emb, cur_emb]

    if FLAGS.use_contexts:
      context_embs = self.context_emb(input['context'])
      # num_contexts = int(len(context_cols) * (len(context_cols) - 1) / 2)
      # context_embs = tf.unstack(context_embs, axis=1)
      # embs += context_embs
      context_emb = self.sum_pooling(context_embs)
      embs += [context_emb]

      item_embs = self.item_emb(input['item'])
      item_emb = self.sum_pooling(item_embs)
      embs += [item_emb]

      cross_embs = self.cross_emb(input['cross'])
      cross_emb = self.sum_pooling(cross_embs)
      embs += [cross_emb]

    embs = tf.stack(embs, axis=1)

    if FLAGS.batch_norm:
      embs = self.batch_norm(embs)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    self.watches = melt.length(input['watch_vids'])
    return self.logit
