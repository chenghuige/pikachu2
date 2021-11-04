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

# uid + docid only 0.5914
class UV(keras.Model):
  def __init__(self):
    super(UV, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    self.dense = keras.layers.Dense(1)

  def call(self, input):
    # gezi.set('input', input)
    uids = input['did']
    vids = input['vid']

    uemb = self.uemb(uids)
    vemb = self.vemb(vids)

    x = tf.concat([uemb, vemb], -1)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# dot 0.7338(train 1 valid 2), cat 0.5924  UVH uvh-dot.sh submit with training on 29,30 got 0.54 only
class UVH(keras.Model):
  def __init__(self):
    super(UVH, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    uemb = self.uemb(input['did'])
    vemb = self.vemb(input['vid'])

    wvembs = self.vemb(input['watch_vids'])

    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))
    
    embs = tf.stack([uemb, vemb, wvemb], axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 0.7402
class Model1(keras.Model):
  def __init__(self):
    super(Model1, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')

    # context
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

    # user info
    remb = self.remb(input['region'])

    # video
    vemb = self.vemb(input['vid'])

    # video info
  
    # context
    #   phone
    embs = [uemb, wvemb, remb, vemb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 0.7411
class Model2(keras.Model):
  def __init__(self):
    super(Model2, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    # video
    vemb = self.vemb(input['vid'])

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pmod_emb, pmf_emb, psver_emb, paver_emb, vemb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 0.78425
class Model3(keras.Model):
  def __init__(self):
    super(Model3, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, vemb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 0.74282
class Model4(keras.Model):
  def __init__(self):
    super(Model4, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, pmod_emb, pmf_emb, psver_emb, paver_emb, vemb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model5(keras.Model):
  def __init__(self):
    super(Model5, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, vemb, vcemb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model6(keras.Model):
  def __init__(self):
    super(Model6, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# Model6 去掉 watch vids
class Model6_1(keras.Model):
  def __init__(self):
    super(Model6_1, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, remb, pemb, 
            vemb, cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model6_2(keras.Model):
  def __init__(self):
    super(Model6_2, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    self.prev_emb = Embedding(10000000, FLAGS.emb_size, num_buckets=1000000, name='prev_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    prev_emb = self.prev_emb(input['prev'])

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# same as 6_2 but share vemb
# share效果较好  添加前片id
class Model6_3(keras.Model):
  def __init__(self):
    super(Model6_3, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model6_4(keras.Model):
  def __init__(self):
    super(Model6_4, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 分开两个doc emb
class Model6_5(keras.Model):
  def __init__(self):
    super(Model6_5, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')
    self.wvemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='wvideo_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.wvemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# smaller vemb -> 50w 
class Model6_6(keras.Model):
  def __init__(self):
    super(Model6_6, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# vid not use hash
class Model6_7(keras.Model):
  def __init__(self):
    super(Model6_7, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    assert not FLAGS.hash_vid
    if FLAGS.use_w2v:
      emb = np.load(FLAGS.vid_w2v_pretrain)
      assert emb.shape[0] == 500000
      print(emb.shape)
      self.vemb = tf.keras.layers.Embedding(500000, emb.shape[1], name='video_emb',
                                            embeddings_initializer=tf.constant_initializer(emb),
                                            trainable=FLAGS.train_emb)
    else:
      self.vemb = keras.layers.Embedding(500000, FLAGS.emb_size, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model6_8(keras.Model):
  def __init__(self):
    super(Model6_8, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    
    assert not FLAGS.hash_vid
    self.vemb = keras.layers.Embedding(500000, FLAGS.emb_size * 2, name='video_emb')
    self.dense_vemb = keras.layers.Dense(FLAGS.emb_size)

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))
    wvemb = self.dense_vemb(wvemb)

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
    vemb = self.dense_vemb(vemb)

    prev_emb = self.vemb(input['prev'])
    prev_emb = self.dense_vemb(prev_emb)

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model6_9(keras.Model):
  def __init__(self):
    super(Model6_9, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    assert not FLAGS.hash_vid
    self.vemb = keras.layers.Embedding(500000, FLAGS.emb_size, name='video_emb')
    
    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
    uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model6_10(keras.Model):
  def __init__(self):
    super(Model6_10, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
    uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# Model6 remove uid
class Model7(keras.Model):
  def __init__(self):
    super(Model7, self).__init__() 

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [wvemb, remb, pemb, vemb, cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model7_1(keras.Model):
  def __init__(self):
    super(Model7_1, self).__init__() 

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [wvemb, 
            vemb, prev_emb,
            remb, pemb, 
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model7_10(keras.Model):
  def __init__(self):
    super(Model7_10, self).__init__() 

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# from 6 with sum vembs and all vembs
class Model8(keras.Model):
  def __init__(self):
    super(Model8, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, vemb, vcemb, cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model9(keras.Model):
  def __init__(self):
    super(Model9, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='user_emb')
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=3000000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [uemb, wvemb, remb, 
            pemb, pmod_emb, pmf_emb, psver_emb, paver_emb,
            vemb, vcemb, cemb, class_emb, second_class_emb, intact_emb]
    
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# from model6_10 adding time info
# 注意eval数据每个用户的时间戳都一样 是否使用hour会有问题 训练集合感觉比较正常 一个用户可能时间跨度7个小时类似 
# 验证数据看 + hour weekday效果不好 那是否还要加weekday 信息？ 是否周末有影响吗 
class Model10(keras.Model):
  def __init__(self):
    super(Model10, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.hour_emb = keras.layers.Embedding(30, FLAGS.emb_size, name='hour_emb')
    self.weekday_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='weekday_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
    uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context

    #   time
    hour_emb = self.hour_emb(input['hour'])
    weekday_emb = self.weekday_emb(input['weekday'])
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            hour_emb, weekday_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit


# 6_10基础上 引入fresh特征 
class Model11(keras.Model):
  def __init__(self):
    super(Model11, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
    uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    fresh_emb = self.fresh_emb(util.get_fresh_intervals(input['fresh']))
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            fresh_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 加入fresh 特征 但是是按照dense加入
class Model11_1(keras.Model):
  def __init__(self):
    super(Model11_1, self).__init__() 

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
    uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    fresh = input['fresh']
    fresh = tf.math.minimum(fresh, 1200)
    fresh = tf.math.maximum(fresh, 0)
    fresh = tf.cast(fresh / 1200, tf.float32)

    dense_feats = tf.stack([fresh], 1)
    dense_emb = self.dense_mlp(dense_feats)
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = tf.concat([dense_emb, x], -1)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit


# 在6-10基础上添加mlp
class Model12(keras.Model):
  def __init__(self):
    super(Model12, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    uemb = self.uemb(input['did'])
    uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [uemb, wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model12_1(keras.Model):
  def __init__(self):
    super(Model12_1, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 在12_1基础上尝试采用全量vid vocab 非hash 并且屏蔽低频转UNK
class Model12_2(keras.Model):
  def __init__(self):
    super(Model12_2, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    assert not FLAGS.hash_vid
    if FLAGS.use_w2v:
      emb = np.load(FLAGS.vid_w2v_pretrain)
      assert emb.shape[0] == 500000
      print(emb.shape)
      self.vemb = tf.keras.layers.Embedding(500000, emb.shape[1], name='video_emb',
                                            embeddings_initializer=tf.constant_initializer(emb),
                                            trainable=FLAGS.train_emb)
    else:
      self.vemb = keras.layers.Embedding(500000, FLAGS.emb_size, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    def _vid(vid):
      return util.get_vid(vid, FLAGS.max_vid)
  
    wvembs = self.vemb(_vid(input['watch_vids']))
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    # video
    vemb = self.vemb(_vid(input['vid']))

    prev_emb = self.vemb(_vid(input['prev']))

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 和12_2相比 采用VocabEmbedding 企图固定低频部分不变 看是否能降低过拟合？
class Model12_3(keras.Model):
  def __init__(self):
    super(Model12_3, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    assert not FLAGS.hash_vid
    assert FLAGS.max_vid
    if FLAGS.use_w2v:
      emb = np.load(FLAGS.vid_w2v_pretrain)
      assert emb.shape[0] == 500000
      print(emb.shape)
      self.vemb = melt.layers.VocabEmbedding(500000, emb.shape[1], name='video_emb',
                                             train_size=FLAGS.max_vid,
                                             embeddings_initializer=tf.constant_initializer(emb),
                                             trainable=FLAGS.train_emb)
    else:
      self.vemb = melt.layers.VocabEmbedding(500000, FLAGS.emb_size, train_size=FLAGS.max_vid, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 无冲突版本qremb 
class Model12_4(keras.Model):
  def __init__(self):
    super(Model12_4, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    assert not FLAGS.hash_vid
    if FLAGS.use_w2v:
      emb = np.load(FLAGS.vid_w2v_pretrain)
      assert emb.shape[0] == 500000
      print(emb.shape)
      self.vemb = melt.layers.MultiplyEmbedding(500000, emb.shape[1], name='video_emb',
                                            embeddings_initializer=tf.constant_initializer(emb),
                                            trainable=FLAGS.train_emb)
    else:
      self.vemb = melt.layers.MultiplyEmbedding(500000, FLAGS.emb_size, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)

    def _vid(vid):
      return util.get_vid(vid, FLAGS.max_vid)
  
    wvembs = self.vemb(_vid(input['watch_vids']))
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

    # user info
    remb = self.remb(input['region'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod'])
    pmf_emb = self.pmf_emb(input['mf'])
    psver_emb = self.psver_emb(input['sver'])
    paver_emb = self.paver_emb(input['aver'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    # video
    vemb = self.vemb(_vid(input['vid']))

    prev_emb = self.vemb(_vid(input['prev']))

    cemb = self.cemb(input['cid'])
    class_emb = self.class_emb(input['class_id'])
    second_class_emb = self.second_class_emb(input['second_class'])
    intact_emb = self.intact_emb(input['is_intact'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# VocabEmb with bias_emb
class Model12_5(keras.Model):
  def __init__(self):
    super(Model12_5, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    assert not FLAGS.hash_vid
    assert FLAGS.max_vid
    if FLAGS.use_w2v:
      emb = np.load(FLAGS.vid_w2v_pretrain)
      assert emb.shape[0] == 500000
      print(emb.shape)
      self.vemb = melt.layers.VocabEmbedding(500000, emb.shape[1], name='video_emb',
                                             train_size=FLAGS.max_vid,
                                             embeddings_initializer=tf.constant_initializer(emb),
                                             bias_emb=True,
                                             trainable=FLAGS.train_emb)
    else:
      self.vemb = melt.layers.VocabEmbedding(500000, FLAGS.emb_size, train_size=FLAGS.max_vid, bias_emb=True, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit


# 在12_1基础上 引入 ctr特征  并且只做dense特征
class Model13_1(keras.Model):
  def __init__(self):
    super(Model13_1, self).__init__() 

    Embedding = melt.layers.QREmbedding

    # TODO remove
    FLAGS.emb_size = 128

    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    ctr = input['ctr']
    dense_feats = tf.stack([ctr], 1)
    dense_emb = self.dense_mlp(dense_feats)
    
    embs = [
            # uemb, 
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 在12_1基础上 引入 ctr特征  并且只加到dot层后面concat
class Model14(keras.Model):
  def __init__(self):
    super(Model14, self).__init__() 

    Embedding = melt.layers.QREmbedding

    # TODO remove
    FLAGS.emb_size = 128

    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    # self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    # dense_feats = tf.stack([ctr], 1)
    # dense_emb = self.dense_mlp(dense_feats)
    
    embs = [
            # uemb, 
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,  
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    ctr = tf.expand_dims(input['ctr'], 1)
    x = tf.concat([x, ctr], 1)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model14_1(keras.Model):
  def __init__(self):
    super(Model14_1, self).__init__() 

    Embedding = melt.layers.QREmbedding

    # TODO remove
    FLAGS.emb_size = 128

    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    # self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    # dense_feats = tf.stack([ctr], 1)
    # dense_emb = self.dense_mlp(dense_feats)
    
    embs = [
            # uemb, 
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,  
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    ctr = input['ctr'] 
    ctrs = tf.stack([ctr, ctr ** 2, ctr ** 0.5], 1)
    x = tf.concat([x, ctrs], axis=1)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model14_2(keras.Model):
  def __init__(self):
    super(Model14_2, self).__init__() 

    Embedding = melt.layers.QREmbedding

    # TODO remove
    FLAGS.emb_size = 128

    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.dense_dense = keras.layers.Dense(FLAGS.emb_size)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    ctr = input['ctr'] 
    ctrs = tf.stack([ctr, ctr ** 2, ctr ** 0.5], 1)
    dense_feats = ctrs
    dense_emb = self.dense_dense(dense_feats)
    
    embs = [
            # uemb, 
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,  
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model14_3(keras.Model):
  def __init__(self):
    super(Model14_3, self).__init__() 

    Embedding = melt.layers.QREmbedding

    # TODO remove
    FLAGS.emb_size = 128

    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.dense_dense = keras.layers.Dense(FLAGS.emb_size)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    ctr = input['ctr'] 
    ctrs = tf.stack([ctr, ctr ** 2, ctr ** 0.5], 1)
    dense_feats = ctrs
    dense_emb = self.dense_dense(dense_feats)
    
    embs = [
            # uemb, 
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,  
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    x = tf.concat([x, dense_emb], -1)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model14_4(keras.Model):
  def __init__(self):
    super(Model14_4, self).__init__() 

    Embedding = melt.layers.QREmbedding

    # TODO remove
    FLAGS.emb_size = 128

    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.fresh_emb = keras.layers.Embedding(10, FLAGS.emb_size, name='fresh_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.dense_dense = keras.layers.Dense(FLAGS.emb_size)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
    ctr = input['ctr'] 
    ctrs = tf.stack([ctr, ctr ** 2, ctr ** 0.5], 1)
    dense_feats = ctrs
    dense_emb = self.dense_mlp(dense_feats)
    
    embs = [
            # uemb, 
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,  
            dense_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    x = tf.concat([x, dense_emb], -1)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 15在 12_1基础上增加stars
class Model15(keras.Model):
  def __init__(self):
    super(Model15, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    # 视频明星
    self.stars_emb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='stars_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,stars_emb,
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 16是在 15基础上的增加title story
class Model16(keras.Model):
  def __init__(self):
    super(Model16, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    # 视频明星
    self.stars_emb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='stars_emb')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = keras.layers.Embedding(1000000, FLAGS.emb_size, name='words_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,stars_emb,
            title_emb, story_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model16_1(keras.Model):
  def __init__(self):
    super(Model16_1, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    # 视频明星
    self.stars_emb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='stars_emb')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='words_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,stars_emb,
            title_emb, story_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# same as 16_1 but vocab bucket from 10w -> 20w
class Model16_2(keras.Model):
  def __init__(self):
    super(Model16_2, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    
    self.vemb = Embedding(40000000, FLAGS.emb_size, num_buckets=1500000, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    # 视频明星
    self.stars_emb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='stars_emb')

    # Compre with qremb or just use compat vocab 167108
    self.words_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=200000, name='words_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,stars_emb,
            title_emb, story_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

#  不再使用hash 全部使用词典
class Model17(keras.Model):
  def __init__(self):
    super(Model17, self).__init__() 

    Embedding = melt.layers.VEmbedding
    vs = gezi.get('vocab_sizes')
    def _emb(vocab_name):
      return Embedding(vs[vocab_name][0], FLAGS.emb_size, vs[vocab_name][1], name=f'{vocab_name}_emb')

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

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,stars_emb,
            title_emb, story_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model17_1(keras.Model):
  def __init__(self):
    super(Model17_1, self).__init__() 

    Embedding = melt.layers.VEmbedding
    vs = gezi.get('vocab_sizes')
    def _emb(vocab_name):
      embeddings_initializer = 'uniform'
      trainable = True
      if FLAGS.use_w2v and vocab_name == 'vid':
        emb = np.load(FLAGS.vid_w2v_pretrain)
        emb = emb[:vs['vid'][0]]
        embeddings_initializer=tf.constant_initializer(emb)
        trainable = FLAGS.train_emb
      return Embedding(vs[vocab_name][0], FLAGS.emb_size, vs[vocab_name][1], 
                       embeddings_initializer=embeddings_initializer, 
                       trainable=trainable, name=f'{vocab_name}_emb')

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

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            # stars_emb,
            # title_emb, story_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# 新tfrecord复现 12-5结果
class Model17_5(keras.Model):
  def __init__(self):
    super(Model17_5, self).__init__() 

    # FLAGS.emb_size = 128

    Embedding = melt.layers.QREmbedding
    # self.uemb = Embedding(40000000, int(FLAGS.emb_size / 2), num_buckets=3000000, name='user_emb')
    # self.dense_uemb = keras.layers.Dense(FLAGS.emb_size)
    vs = gezi.get('vocab_sizes')
    train_size = FLAGS.max_vid
    if FLAGS.use_w2v:
      emb = np.load(FLAGS.vid_w2v_pretrain)
      emb = emb[:vs['vid'][0]]
      self.vemb = melt.layers.VocabEmbedding(vs['vid'][0], emb.shape[1], name='video_emb',
                                             train_size=train_size,
                                             embeddings_initializer=tf.constant_initializer(emb),
                                             trainable=FLAGS.train_emb)
    else:
      self.vemb = melt.layers.VocabEmbedding(vs['vid'][0], FLAGS.emb_size, train_size=train_size, name='video_emb')

    # user context
    self.remb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='region_emb')
    #   phone
    self.pmod_emb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='phone_mod_emb')
    self.pmf_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_mf_emb')
    self.psver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_sver_emb')
    self.paver_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='phone_aver_emb')

    # 视频所属合集 
    self.cemb = Embedding(3000000, FLAGS.emb_size, num_buckets=300000, name='cid_emb')
    # 视频类别
    self.class_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='class_emb')
    self.second_class_emb = Embedding(500000, FLAGS.emb_size, num_buckets=50000, name='second_class_em')
    self.cemb = Embedding(1000000, FLAGS.emb_size, num_buckets=100000, name='cid_emb')
    self.intact_emb = Embedding(100000, FLAGS.emb_size, num_buckets=10000, name='intact_emb')

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

    # user info
    remb = self.remb(input['region_'])
    #   phone
    pmod_emb = self.pmod_emb(input['mod_'])
    pmf_emb = self.pmf_emb(input['mf_'])
    psver_emb = self.psver_emb(input['sver_'])
    paver_emb = self.paver_emb(input['aver_'])

    pemb = pmod_emb + pmf_emb + psver_emb + paver_emb

    # video
    vemb = self.vemb(input['vid'])

    prev_emb = self.vemb(input['prev'])

    cemb = self.cemb(input['cid_'])
    class_emb = self.class_emb(input['class_id_'])
    second_class_emb = self.second_class_emb(input['second_class_'])
    intact_emb = self.intact_emb(input['is_intact_'])

    vcemb = cemb + class_emb + second_class_emb + intact_emb

    # video info
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)

    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# online 0.655
class Model17_2(keras.Model):
  def __init__(self):
    super(Model17_2, self).__init__() 

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

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# add video ctr
class Model17_3(keras.Model):
  def __init__(self):
    super(Model17_3, self).__init__() 
    
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

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.dense_dense = keras.layers.Dense(FLAGS.emb_size)

    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    ctr = input['ctr'] 
    ctrs = tf.stack([ctr, ctr ** 2, ctr ** 0.5], 1)
    dense_feats = ctrs
    dense_emb = self.dense_dense(dense_feats)
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

# add all dense video feats
class Model17_4(keras.Model):
  def __init__(self):
    super(Model17_4, self).__init__() 
    
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

    self.dense = keras.layers.Dense(1)

    self.sum_pooling = melt.layers.SumPooling()

    self.pooling = melt.layers.Pooling(FLAGS.pooling)

    # TODO 尝试扩大这两个mlp 参考dlrm
    # --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size, FLAGS.emb_size])
    self.dense_dense = keras.layers.Dense(FLAGS.emb_size)
    self.mlp = melt.layers.MLP([128, 32], name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    ctr = melt.scalar_feature(input['ctr'] )
    vv = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = melt.scalar_feature(input['duration'], max_val=10000, scale=True)
    title_len = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh = melt.scalar_feature(fresh, max_val=1200, scale=True)
    dense_feats = tf.concat([ctr, vv, vdur, title_len, fresh], -1)

    if not FLAGS.dense_mlp:
      dense_emb = self.dense_dense(dense_feats)
    else:
      dense_emb = self.dense_mlp(dense_feats)
    
    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

class Model17_6(keras.Model):
  def __init__(self):
    super(Model17_6, self).__init__() 
    
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
    self.dense_mlp = melt.layers.MLP([FLAGS.emb_size * 2, FLAGS.emb_size], activation='relu', name='dense_mlp')
    self.mlp = melt.layers.MLP([256, 128, 32], activation='relu', name='mlp')

  def call(self, input):
    # user 
    # uemb = self.uemb(input['did'])
    # uemb = self.dense_uemb(uemb)
  
    wvembs = self.vemb(input['watch_vids'])
    wvemb = self.sum_pooling(wvembs, melt.length(input['watch_vids']))

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

    story_embs = self.words_emb(input['story'])
    story_emb = self.sum_pooling(story_embs, melt.length(input['story']))
  
    # context
    ctr = melt.scalar_feature(input['ctr'] )
    vv = melt.scalar_feature(input['vv'], max_val=100000, scale=True)
    vdur = melt.scalar_feature(input['duration'], max_val=10000, scale=True)
    title_len = melt.scalar_feature(tf.cast(input['title_length'], tf.float32), max_val=205, scale=True)
    fresh = tf.cast(input['fresh'], tf.float32) / (3600 * 24)
    fresh = melt.scalar_feature(fresh, max_val=1200, scale=True)
    dense_feats = tf.concat([ctr, vv, vdur, title_len, fresh], -1)
    dense_emb = self.dense_mlp(dense_feats)

    embs = [
            wvemb, remb, pemb, 
            vemb, prev_emb,
            cemb, class_emb, second_class_emb, intact_emb,
            stars_emb,
            title_emb, story_emb,
            dense_emb
            ]
    embs = tf.stack(embs, axis=1)

    x = self.pooling(embs)
    x = tf.concat([x, dense_emb], axis=1)
    x = self.mlp(x)

    self.logit = self.dense(x)
    self.prob = tf.math.sigmoid(self.logit)
    self.index = input['index']
    return self.logit

