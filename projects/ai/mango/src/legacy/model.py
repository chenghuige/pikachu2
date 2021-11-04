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

# Base is simple and fast 没有用image特征 lstm等等 
class Base(keras.Model):
  def __init__(self):
    super(Base, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_encoder = None
    if FLAGS.history_encoder in ['lstm', 'gru']:
      self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers,
                                      num_units=int(FLAGS.hidden_size / 2),
                                      keep_prob=1. - FLAGS.dropout,
                                      share_dropout=False,
                                      recurrent_dropout=False,
                                      concat_layers=FLAGS.concat_layers,
                                      bw_dropout=False,
                                      residual_connect=False,
                                      train_init_state=False,
                                      cell=FLAGS.history_encoder)
      # self.his_dense = keras.layers.Dense(FLAGS.emb_size)
      self.his_dense = None
    elif FLAGS.history_encoder in ['LSTM', 'GRU']:
      Encoder = getattr(tf.keras.layers, FLAGS.history_encoder)
      self.his_encoder = Encoder(FLAGS.hidden_size, return_sequences=True,
                                 dropout=FLAGS.dropout, recurrent_dropout=FLAGS.rdropout)
      self.his_dense = None

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.deal_history(wvembs, melt.length(input['watch_vids']))

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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    dense_emb = self.deal_dense(input)

    embs = [
            wvemb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

# Current STOA
class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh  历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    his_emb = self.sum_pooling(self.his_encoder(his_embs, wv_len), wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class Model2(keras.Model):
  def __init__(self):
    super(Model2, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh  历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    cur_embs = tf.expand_dims(tf.concat([vemb, title_emb], axis=-1), 1)
    his_embs = tf.concat([cur_embs, his_embs], axis=1)
    his_emb = self.sum_pooling(self.his_encoder(his_embs, wv_len + 1), wv_len + 1)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

# use first pooling
class Model2_1(keras.Model):
  def __init__(self):
    super(Model2_1, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh  历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    cur_embs = tf.expand_dims(tf.concat([vemb, title_emb], axis=-1), 1)
    his_embs = tf.concat([cur_embs, his_embs], axis=1)
    his_emb = melt.layers.FirstPooling()(self.his_encoder(his_embs, wv_len + 1))

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelTransformer(keras.Model):
  def __init__(self):
    super(ModelTransformer, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.transformer.Encoder(num_layers=3, d_model=FLAGS.hidden_size * 2, num_heads=4, 
                                                       dff=FLAGS.hidden_size * 2, maximum_position_encoding=100,
                                                       rate=FLAGS.dropout)
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh  历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    cur_embs = tf.expand_dims(tf.concat([vemb, title_emb], axis=-1), 1)
    his_embs = tf.concat([cur_embs, his_embs], axis=1)
    his_emb = melt.layers.FirstPooling()(self.his_encoder(his_embs, wv_len + 1))
    his_emb = self.his_dense(his_emb)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class BaseMultiHead(keras.Model):
  def __init__(self):
    super(BaseMultiHead, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.vid_pooling = melt.layers.MultiHeadAttention(num_heads=4)

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes TODO remove - with new dataset
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    # video
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    wvemb = self.vid_pooling(vemb, wvembs, wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    dense_emb = self.deal_dense(input)

    embs = [
            wvemb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class BaseDin(keras.Model):
  def __init__(self):
    super(BaseDin, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_pooling = melt.layers.DinAttention()

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes TODO remove - with new dataset
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    # video
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    wvemb = self.his_pooling(vemb, wvembs, wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    dense_emb = self.deal_dense(input)

    embs = [
            wvemb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class BaseDinImg(keras.Model):
  def __init__(self):
    super(BaseDinImg, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    self.image_emb = _emb('image')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_pooling = melt.layers.DinAttention()
    self.his_img_pooling = melt.layers.DinAttention()

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes TODO remove - with new dataset
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    # video
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    wvemb = self.his_pooling(vemb, wvembs, wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.his_img_pooling(image_emb, his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    dense_emb = self.deal_dense(input)

    embs = [
            wvemb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            image_emb,
            his_image_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class BaseDinImg2(keras.Model):
  def __init__(self):
    super(BaseDinImg2, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    self.image_emb = _emb('image')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_pooling = melt.layers.DinAttention()
    self.his_img_pooling = melt.layers.DinAttention()

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes TODO remove - with new dataset
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    # video
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    wvemb = self.his_pooling(vemb, wvembs, wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_embs = self.image_mlp(his_image_embs)
    his_image_emb = self.his_img_pooling(image_emb, his_image_embs, wv_len)
    
    dense_emb = self.deal_dense(input)

    embs = [
            wvemb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            image_emb,
            his_image_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit


class BaseDin2(keras.Model):
  def __init__(self):
    super(BaseDin2, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.his_pooling = melt.layers.DinAttention()
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes TODO remove - with new dataset
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    # video
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    
    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])
    his_embs = tf.concat([wvembs, cembs, class_embs, second_class_embs, intact_embs], axis=-1)
    cur_emb = tf.concat([vemb, cemb, class_emb, second_class_emb, intact_emb], axis=-1)
    his_emb = self.his_pooling(cur_emb, his_embs, wv_len)
    his_emb = self.his_dense(his_emb)

    dense_emb = self.deal_dense(input)

    embs = [
            his_emb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class BaseMhead2(keras.Model):
  def __init__(self):
    super(BaseMhead2, self).__init__()

    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集
    self.cemb = _emb('aver')
    # 视频类别
    self.class_emb = _emb('class_id')
    self.second_class_emb = _emb('second_class')
    self.cemb = _emb('cid')
    self.intact_emb = _emb('is_intact')

    # 视频明星
    self.stars_emb = _emb('stars')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = _emb('words')

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                      activation=activation, name='dense_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    self.vid_pooling = melt.layers.MultiHeadAttention(num_heads=4)
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, wvembs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      wvembs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(wvembs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)

    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)

    # 用户历史平均fresh 历史展现时间戳有 历史video发布时间戳暂时没有 freshes TODO remove - with new dataset
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_,
                              num_stars_, fresh_, num_hists_,
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    # video
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    
    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))

    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])
    his_embs = tf.concat([wvembs, cembs, class_embs, second_class_embs, intact_embs], axis=-1)
    cur_emb = tf.concat([vemb, cemb, class_emb, second_class_emb, intact_emb], axis=-1)
    his_emb = self.his_pooling(cur_emb, his_embs, wv_len)
    his_emb = self.his_dense(his_emb)

    dense_emb = self.deal_dense(input)

    embs = [
            his_emb, remb, pemb,
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            titles_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

# lstm-dim now best
class ModelDin(keras.Model):
  def __init__(self):
    super(ModelDin, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    his_emb = self.his_encoder(his_embs, wv_len)
    cur_emb = self.cur_dense(tf.concat([vemb, title_emb], axis=-1))
    his_emb = self.his_pooling(cur_emb, his_emb, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelDin_1(keras.Model):
  def __init__(self):
    super(ModelDin_1, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    his_emb = self.his_encoder(his_embs, wv_len)
    cur_emb = self.cur_dense(tf.concat([vemb, title_emb], axis=-1))
    his_emb = self.his_pooling(cur_emb, his_emb, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_embs = self.image_mlp(his_image_embs)
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    
    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelDin2(keras.Model):
  def __init__(self):
    super(ModelDin2, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'])
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])
    vcembs = cembs + class_embs + second_class_embs + intact_embs

    his_embs = tf.concat([wvembs, titles_embs, vcembs], axis=-1)
    his_emb = self.his_encoder(his_embs, wv_len)
    cur_emb = self.cur_dense(tf.concat([vemb, title_emb, vcemb], axis=-1))
    his_emb = self.his_pooling(cur_emb, his_emb, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb, vcemb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelDin3(keras.Model):
  def __init__(self):
    super(ModelDin3, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.his_title_pooling = melt.layers.DinAttention()
    self.his_class_pooling = melt.layers.DinAttention()
    self.his_stars_pooling = melt.layers.DinAttention()
    self.his_image_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'])
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
    batch_size = melt.get_shape(input['vid'], 0)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vc_emb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    stars_list_embs = self.stars_emb(input['stars_list'])
    stars_list_embs = tf.reshape(stars_list_embs, [-1, 10, FLAGS.hidden_size])
    stars_list_embs = self.sum_pooling(stars_list_embs)
    stars_list_embs = tf.reshape(stars_list_embs, [batch_size, -1, FLAGS.hidden_size])
    stars_list_emb = self.his_stars_pooling(stars_emb, stars_list_embs, wv_len)

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])
    titles_emb = self.his_title_pooling(title_emb, titles_embs, wv_len)

    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])
    vc_embs = cembs + class_embs + second_class_embs + intact_embs

    vcs_emb = self.his_class_pooling(vc_emb, vc_embs, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_embs = self.image_mlp(his_image_embs)
    his_image_emb = self.his_image_pooling(image_emb, his_image_embs, wv_len)

    his_embs = tf.concat([wvembs, titles_embs, vc_embs, stars_list_embs], axis=-1)
    his_emb = self.his_encoder(his_embs, wv_len)
    cur_emb = self.cur_dense(tf.concat([vemb, title_emb, vc_emb, stars_emb], axis=-1))
    his_emb = self.his_pooling(cur_emb, his_emb, wv_len)
    
    dense_emb = self.deal_dense(input)
    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb, 
            vc_emb,
            vcs_emb,
            stars_emb,
            stars_list_emb,
            title_emb, story_emb, 
            titles_emb,
            image_emb,
            his_image_emb,
            his_emb,
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelDin4(keras.Model):
  def __init__(self):
    super(ModelDin4, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.his_dense = keras.layers.Dense(FLAGS.hidden_size)
    self.his_title_pooling = melt.layers.DinAttention()
    self.his_class_pooling = melt.layers.DinAttention()
    self.his_stars_pooling = melt.layers.DinAttention()
    self.his_image_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'])
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
    batch_size = melt.get_shape(input['vid'], 0)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vc_emb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    stars_list_embs = self.stars_emb(input['stars_list'])
    stars_list_embs = tf.reshape(stars_list_embs, [-1, 10, FLAGS.hidden_size])
    stars_list_embs = self.sum_pooling(stars_list_embs)
    stars_list_embs = tf.reshape(stars_list_embs, [batch_size, -1, FLAGS.hidden_size])
    # stars_list_emb = self.his_stars_pooling(stars_emb, stars_list_embs, wv_len)

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])
    # titles_emb = self.his_title_pooling(title_emb, titles_embs, wv_len)

    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])
    vc_embs = cembs + class_embs + second_class_embs + intact_embs

    # vcs_emb = self.his_class_pooling(vc_emb, vc_embs, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_embs = self.image_mlp(his_image_embs)
    # his_image_emb = self.his_image_pooling(image_emb, his_image_embs, wv_len)

    his_embs = tf.concat([wvembs, titles_embs, vc_embs, stars_list_embs, his_image_embs], axis=-1)
    his_emb = self.his_encoder(his_embs, wv_len)
    cur_emb = self.cur_dense(tf.concat([vemb, title_emb, vc_emb, stars_emb, image_emb], axis=-1))
    his_emb = self.his_pooling(cur_emb, his_emb, wv_len)
    
    dense_emb = self.deal_dense(input)
    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb, 
            vc_emb,
            # vcs_emb,
            stars_emb,
            # stars_list_emb,
            title_emb, story_emb, 
            # titles_emb,
            image_emb,
            # his_image_emb,
            his_emb,
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelDin5(keras.Model):
  def __init__(self):
    super(ModelDin5, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
    vemb = self.vemb(input['vid'])
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    wvemb = self.his_pooling(vemb, self.his_encoder(wvembs, wv_len), wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    # video
    

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    # his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    # his_emb = self.his_encoder(his_embs, wv_len)
    # cur_emb = self.cur_dense(tf.concat([vemb, title_emb], axis=-1))
    # his_emb = self.his_pooling(cur_emb, his_emb, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, wvemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            # his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelDin6(keras.Model):
  def __init__(self):
    super(ModelDin6, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.DinAttention()
    self.his_vid_pooling = melt.layers.DinAttention()
    self.his_title_pooling = melt.layers.DinAttention()
    self.his_class_pooling = melt.layers.DinAttention()
    self.his_stars_pooling = melt.layers.DinAttention()
    self.his_image_pooling = melt.layers.DinAttention()
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
    batch_size = melt.get_shape(input['vid'], 0)
  
    vemb = self.vemb(input['vid'])
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
    wvemb = self.his_vid_pooling(vemb, self.his_encoder(wvembs, wv_len), wv_len)

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    # video
    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vc_emb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    stars_list_embs = self.stars_emb(input['stars_list'])
    stars_list_embs = tf.reshape(stars_list_embs, [-1, 10, FLAGS.hidden_size])
    stars_list_embs = self.sum_pooling(stars_list_embs)
    stars_list_embs = tf.reshape(stars_list_embs, [batch_size, -1, FLAGS.hidden_size])
    stars_list_emb = self.his_stars_pooling(stars_emb, stars_list_embs, wv_len)

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])
    titles_emb = self.his_title_pooling(title_emb, titles_embs, wv_len)

    cembs = self.cemb(input['cids'])
    class_embs = self.class_emb(input['class_ids'])
    second_class_embs = self.second_class_emb(input['second_classes'])
    intact_embs = self.intact_emb(input['is_intacts'])
    vc_embs = cembs + class_embs + second_class_embs + intact_embs

    vcs_emb = self.his_class_pooling(vc_emb, vc_embs, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_embs = self.image_mlp(his_image_embs)
    his_image_emb = self.his_image_pooling(image_emb, his_image_embs, wv_len)

    # his_embs = tf.concat([wvembs, titles_embs, vc_embs, stars_list_embs], axis=-1)
    # his_emb = self.his_encoder(his_embs, wv_len)
    # cur_emb = self.cur_dense(tf.concat([vemb, title_emb, vc_emb, stars_emb], axis=-1))
    # his_emb = self.his_pooling(cur_emb, his_emb, wv_len)
    
    dense_emb = self.deal_dense(input)
    embs = [
            remb, pemb, 
            vemb, wvemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb, 
            vc_emb,
            vcs_emb,
            stars_emb,
            stars_list_emb,
            title_emb, story_emb, 
            titles_emb,
            image_emb,
            his_image_emb,
            # his_emb,
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit

class ModelMHead(keras.Model):
  def __init__(self):
    super(ModelMHead, self).__init__() 
    
    def _emb(vocab_name):
      return util.create_emb(vocab_name)

    self.vemb = _emb('vid')
    # region
    self.remb = _emb('region')
    #   phone
    self.pmod_emb = _emb('mod')
    self.pmf_emb = _emb('mf')
    self.psver_emb = _emb('sver')
    self.paver_emb = _emb('aver')

    # 视频所属合集 
    self.cemb = _emb('aver')
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

    self.sum_pooling = melt.layers.SumPooling()
    # self.mean_pooling = melt.layers.MeanPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense = keras.layers.Dense(1)
    # TODO 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    activation = FLAGS.activation
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], 
                                      activation=activation, name='dense_mlp')
    self.image_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size],
                                     activation=activation,name='image_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation=activation,
                               drop_rate=FLAGS.mlp_dropout, name='mlp')

    
    self.his_encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=FLAGS.history_encoder)

    self.his_pooling = melt.layers.MultiHeadAttention(num_heads=4)
    self.cur_dense = keras.layers.Dense(FLAGS.hidden_size)

  def deal_history(self, embs, length=None):
    if self.his_encoder is not None:
      # wvembs = self.his_encoder(wvembs, length)
      embs = self.his_encoder(wvembs)
      # TODO why CudnnRnn with hidden size 128 not work... out 256 then turn back to 128 using dense 可能是过拟合 现在改小lr 可以再实验一下
      # if self.his_dense is not None:
      #   wvembs = self.his_dense(wvembs)
    return self.sum_pooling(embs, length)

  def deal_dense(self, input):
    ctr_ = melt.scalar_feature(input['ctr'] )
    vv_ = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = input['duration']
    vdur_ = melt.scalar_feature(vdur, max_val=10000, scale=True)
    title_len_ = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    twords_len_ = melt.scalar_feature(tf.cast(melt.length(input['title']), tf.float32), max_val=40, scale=True)
    num_stars_ = melt.scalar_feature(tf.cast(melt.length(input['stars']), tf.float32), max_val=34, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh_ = melt.scalar_feature(fresh, max_val=1200, scale=True)
    # 用户阅读历史个数 表示一定用户活跃度 比较重要
    num_hists_ = melt.scalar_feature(tf.cast(melt.length(input['watch_vids']), tf.float32), max_val=50, scale=True)
    
    # 用户历史平均dur  好像影响不大 但是也还是new vid效果好一些
    durs = input['durations']
    avg_durs = tf.reduce_sum(tf.math.minimum(durs, 10000), 1) / (tf.cast(melt.length(durs), tf.float32) + 0.00001)
    avg_durs_ = melt.scalar_feature(avg_durs, max_val=10000, scale=True)

    delta_durs = tf.math.maximum(vdur - avg_durs, 0.)
    delta_durs_ = melt.scalar_feature(delta_durs, max_val=10000, scale=True)
    
    # 用户历史平均fresh 需要重新做数据 从record读取 历史展现时间戳有 历史video发布时间戳暂时没有 freshes
    freshes = -input['freshes'] / (3600 * 24)
    avg_freshes = tf.reduce_sum(tf.math.minimum(freshes, 1200), 1) / (tf.cast(melt.length(input['freshes']), tf.float32) + 0.00001)
    avg_freshes_ = melt.scalar_feature(avg_freshes, max_val=1200, scale=True)
    delta_freshes = tf.math.maximum(fresh - avg_freshes, 0.)
    delta_freshes_ = melt.scalar_feature(delta_freshes, max_val=1200, scale=True)

    dense_feats = tf.concat([
                              ctr_, vv_, vdur_, title_len_, twords_len_, 
                              num_stars_, fresh_, num_hists_, 
                              avg_durs_, delta_durs_,
                              avg_freshes_, delta_freshes_,
                            ], -1)
    dense_emb = self.dense_mlp(dense_feats)
    return dense_emb

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wv_len = melt.length(input['watch_vids'])
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

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info

    stars_embs = self.stars_emb(input['stars'])
    stars_emb =  self.sum_pooling(stars_embs, melt.length(input['stars']))

    title_embs = self.words_emb(input['title'])
    title_emb = self.sum_pooling(title_embs, melt.length(input['title']))
    # title_embs = self.title_encoder(title_embs)
    # title_emb = self.sum_pooling(title_embs, melt.length(input['title']))

    titles_embs = self.words_emb(input['titles'])
    batch_size = melt.get_shape(input['vid'], 0)
    titles_embs = tf.reshape(titles_embs, [-1, 10, FLAGS.hidden_size])
    titles_embs = self.sum_pooling(titles_embs)
    titles_embs = tf.reshape(titles_embs, [batch_size, -1, FLAGS.hidden_size])

    his_embs = tf.concat([wvembs, titles_embs], axis=-1)
    his_emb = self.his_encoder(his_embs, wv_len)
    cur_emb = self.cur_dense(tf.concat([vemb, title_emb], axis=-1))
    his_emb = self.his_pooling(cur_emb, his_emb, wv_len)

    # titles_embs = self.title_encoder(titles_embs)
    # titles_emb = self.sum_pooling(title_embs)

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    dense_emb = self.deal_dense(input)

    # image_emb = input['image_emb']
    # image_emb = tf.reshape(image_emb, [-1, 128])
    image_emb = self.image_emb(input['vid'])
    image_emb = self.image_mlp(image_emb)

    his_image_embs = self.image_emb(input['watch_vids'])
    his_image_emb = self.sum_pooling(his_image_embs, melt.length(input['watch_vids']))
    his_image_emb = self.image_mlp(his_image_emb)

    embs = [
            remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb, 
            his_emb,
            dense_emb,
            image_emb,
            his_image_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    self.did = input['did_']
    self.vid = input['vid_']
    return self.logit
