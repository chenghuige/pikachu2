#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2020-05-28 16:18:08.729010
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import gezi 
logging = gezi.logging
import melt
from projects.ai.mind.src.config import *

# for others got from model
def out_hook(model): 
  return dict(prob=model.prob, impression_id=model.impression_id, 
              position=model.position, history_len=model.history_len)

def write_result(ids, predicts, ofile, others, is_infer=True):
  impression_id = others['impression_id']
  prob = others['prob']
  position = others['position']
  df = pd.DataFrame({'impression_id': impression_id, 'position': position, 'score': prob})
  df = df.sort_values(['impression_id', 'position'])
  df = df.groupby('impression_id')['score'].apply(list).reset_index(name='scores')
  df.to_csv(ofile, index=False, header=False, sep=' ')
  df.scores = df.scores.apply(lambda x:  '[' + ','.join(map(str,(-np.asarray(x)).argsort().argsort() + 1)) + ']')
  if is_infer:
    odir = os.path.dirname(ofile)
    df.to_csv(f'{odir}/prediction.txt', index=False, header=False, sep=' ')
    os.system(f'cd {odir};zip prediction.zip prediction.txt')  

def infer_write(ids, predicts, ofile, others):
  return write_result(ids, predicts, ofile, others, is_infer=True)

def valid_write(ids, labels, predicts, ofile, others):
    return write_result(ids, predicts, ofile, others, is_infer=False)

# 1天以内，3天以内，7天以内，30天以内，1年以内，3年以内，超过3年
def get_fresh_interval(timespan):
  days = timespan / (3600 * 24)
  intervals = [1,3,7,30,365,1200]
  for i, item in enumerate(intervals):
    if days <= item:
      return i
  return i + 1

def _get_fresh_intervals(timespans):
  return np.asarray([get_fresh_interval(x) for x in timespans])

def get_fresh_intervals(x):
  res = tf.numpy_function(_get_fresh_intervals, [x], tf.int64)
  res.set_shape(x.get_shape())
  return res

# mark low freq vid as unk_id=1
def get_id(id, max_id, unk_id=1):
  if not max_id:
    return id
  small_id_mask = tf.cast(id <= max_id, tf.int64)
  large_id_mask = 1 - small_id_mask
  final_id = id * small_id_mask + unk_id * large_id_mask
  return final_id

def mask_dids(dids, dids_in_train, subset, test_all_mask=False, unk_id=1):
  need_mask_dids = (subset == 'valid' and FLAGS.valid_mask_dids) \
                    or (subset == 'train' and FLAGS.train_mask_dids) \
                    or (subset == 'test' and FLAGS.test_mask_dids)
  if not need_mask_dids:
    return dids
  
  shape = tf.shape(dids)
  if subset == 'train':
    rand_mask = tf.cast(tf.random.uniform(shape)  > FLAGS.mask_dids_ratio, dids.dtype)
    return dids * rand_mask + unk_id * (1 - rand_mask)
  elif subset == 'valid':
    # dev new did ratio 0.054, test is 0.872  0.874 / (1 - 0.054) = 0.92
    # 采用这个策略dev验证结果和线上test结果基本一致 不过设置了seed=1024好像每次验证还是有一点点指标变化 同样100万验证 不做valid_mask_dids每次结果一致
    mask_dids_ratio = 0.92 
    rand_mask = tf.cast(tf.random.uniform(shape, seed=1024)  > mask_dids_ratio, dids.dtype)
    dids = dids * rand_mask + unk_id * (1 - rand_mask)
    # -- 0.6969 完全随机mask掉92%到0  对应验证test做mask更加一致
    return dids 
    ## -- 0.6969 如果不在train保持不变(5%)，否则随机92%的概率变成0  对应test不做mask更加结果一致
    # new_dids = dids * (1 - dids_in_train)
    # dids += new_dids
    # return dids
    ## 如果全0 --0.6964 不过只用了100万验证 大概结论
    # return tf.zeros_like(dids)
  else: # test 默认 FLAGS.test_mask_ids == False did都保持不变 随机 或者 少量在train里面(13%)经过训练 
    if not test_all_mask:
      # 在train里面的保持不变(13%) 其它的1，如果考虑利用did的一些信息这样是更好 但是和下面全部取1对结果影响应该不大
      return dids * dids_in_train + unk_id * (1 - dids_in_train)
    else:
      # 全部取 1
      return tf.ones_like(dids) * unk_id

def mask_uids(uids, training=False, unk_id=1):
  if not training or not FLAGS.mask_uids_ratio:
    return uids

  rand_mask = tf.cast(tf.random.uniform(tf.shape(uids))  > FLAGS.mask_uids_ratio, uids.dtype)
  return uids * rand_mask + unk_id * (1 - rand_mask)

def unk_aug(x, training=False, x_mask=None, unk_id=1):
  """
  randomly make x% words as unk
  """
  # if not self.training or not FLAGS.unk_aug or melt.epoch() < FLAGS.unk_aug_start_epoch:
  #   return x 
  if not training or not FLAGS.unk_aug_rate:
    return x
    
  if x_mask is None:
    x_mask = x > 0
  x_mask = tf.cast(x_mask, dtype=tf.int64)
  ratio = tf.random.uniform([1,], 0, FLAGS.unk_aug_rate)
  mask = tf.cast(tf.random.uniform(tf.shape(x))  > ratio, x.dtype)
  rmask = unk_id * (1 - mask)
  x = (x * mask + rmask) * x_mask
  return x

def mask_negative_weights(features, training=False):
  ratio = FLAGS.neg_mask_ratio
  if ratio > 0:
    if 'weight' not in features:
      features['weight'] = tf.ones_like(features['did'], dtype=tf.float32)
    if training:
      # mask = tf.cast(tf.random.uniform((melt.get_shape(features['weight'], 0), 1)) > ratio, features['weight'].dtype)
      mask = tf.cast(tf.random.uniform(tf.shape(features['weight'])) > ratio, features['weight'].dtype)
      # mask = tf.squeeze(mask, -1)
      click = tf.cast(features['click'], tf.float32)
      # 注意不是精确的对负样本 采样10% mask 而是整个batch 采样10% 但是如果正样本 不会被mask 效果差不多 后面也可以直接生成tfrecord时候 负样本rand一个数字 0-99 这样可以根据数字过滤 
      features['weight'] = tf.cast((1. - click) * mask, tf.float32) * features['weight'] + click * features['weight'] 
  return features

def adjust(features, subset):
  if 'hist_len' not in features:
    try:
      features['hist_len'] = melt.length(features['history'])
    except Exception:
      features['hist_len'] = tf.ones_like(features['did'])

  if FLAGS.max_history:
    for key in features:
      if key.startswith('history'):
        max_history = FLAGS.max_history
        if 'entity' in key:
          max_history *= 2
        features[key] = features[key][:,:max_history]

  # 注意按照nid去获取新闻测信息 did只是用作id特征 可能被mask
  features['ori_did'] = features['did'] 
  features['ori_history'] = features['history']
  if 'impressions' in features:
    features['ori_impressions'] = features['impressions']

  features['did'] = mask_dids(features['did'], features['did_in_train'],
                              subset, FLAGS.test_all_mask)
  features['uid'] = mask_uids(features['uid'], subset=='train')

  try:
    features['history'] = unk_aug(features['history'], subset=='train')
  except Exception:
    pass
  mask_negative_weights(features, subset=='train')

  if FLAGS.min_count_unk and FLAGS.min_count:
    vs = gezi.get('vocab_sizes')
    features['uid'] = get_id(features['uid'], vs['uid'][1])
    features['did'] = get_id(features['did'], vs['did'][1])
    if FLAGS.mask_history:
      features['history'] = get_id(features['history'], vs['did'][1])
    if 'impressins' in features:
      features['impressions'] = get_id(features['impressions'], vs['did'][1])

  return features

def create_emb(vocab_name, emb_name=None):
  if emb_name is None:
    emb_name = vocab_name
  
  vs = gezi.get('vocab_sizes')
  
  # Embedding = melt.layers.VEmbedding if FLAGS.use_vocab_emb else keras.layers.Embedding
  Embedding = melt.layers.PrEmbedding
  embeddings_initializer = 'uniform'
  trainable = True
  
  if vocab_name == 'uid':
    trainable = FLAGS.train_uid_emb

  if vocab_name == 'did':
    trainable = FLAGS.train_did_emb

  kwargs = {}

  def _load_emb(name, path, trainable):
    emb = np.load(path)
    logging.info(name, emb.shape, vs[name][0])
    assert emb.shape[0] == vs[name][0], f'{emb.shape[0]} {vs[name][0]}'
    # emb = emb[:vs[name][0]]
    embeddings_initializer=tf.constant_initializer(emb)
    # vs[name][0] = emb.shape[0] # change here to reset if vocab size larger then emb height
    # This is expected behavior. trainable=False is a keras concept; it doesn't automatically override the graph and op behavior of tf 1.x. 
    # In order to do what you want, you should use the new style tf.keras.optimizers optimizers (or GradientTape) 
    # which accept a list of variables to differentiate with respect to and the .trainable_weights attribute of Layers and Models which will filter based on .trainable.
    # TODO seems PrEmbedding is slow so just use Embedding 用户Embedding trainable再对应非keras模式keras optimizer会失效 False设置不起作用
    Embedding = melt.layers.PrEmbedding
    kwargs['base_dim'] = emb.shape[1]
    # Embedding = keras.layers.Embedding
    return Embedding, embeddings_initializer, trainable

  if FLAGS.use_entity_pretrain and vocab_name == 'entity':
    Embedding, embeddings_initializer, trainable = _load_emb('entity', FLAGS.entity_pretrain, FLAGS.train_entity_emb)
  if FLAGS.use_word_pretrain and vocab_name == 'word':
    Embedding, embeddings_initializer, trainable = _load_emb('word', FLAGS.word_pretrain, FLAGS.train_word_emb)
  if FLAGS.use_did_pretrain and vocab_name == 'did':
    Embedding, embeddings_initializer, trainable = _load_emb('did', FLAGS.did_pretrain, FLAGS.train_did_emb)

  emb_height = vs[vocab_name][0] if not FLAGS.slim_emb_height else vs[vocab_name][1]
  logging.info(vocab_name, vs[vocab_name][0], vs[vocab_name][1], emb_height, FLAGS.emb_size, embeddings_initializer, trainable)
  return Embedding(emb_height, FLAGS.emb_size, 
              embeddings_initializer=embeddings_initializer, 
              trainable=trainable, 
              name=f'{emb_name}_emb',
              **kwargs)

def create_image_emb():
  emb = np.load('../input/all/image_emb.npy')
  embeddings_initializer=tf.constant_initializer(emb)
  trainable = FLAGS.train_image_emb
  trainable = False
  return tf.keras.layers.Embedding(emb.shape[0], emb.shape[1], embeddings_initializer, 
                                   trainable=trainable, name='image_emb')

def get_encoder(encoder):
  if encoder:
    encoder = melt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
                                    num_units=int(FLAGS.hidden_size / 2), 
                                    keep_prob=1. - FLAGS.dropout,
                                    share_dropout=False,
                                    recurrent_dropout=False,
                                    concat_layers=FLAGS.concat_layers,
                                    bw_dropout=False,
                                    residual_connect=False,
                                    train_init_state=False,
                                    cell=encoder)
  else:
    encoder = lambda x, y=None: x
  return encoder

def get_att_pooling(pooling):
  if pooling == 'din':
    att_activation = FLAGS.att_activation
    if att_activation == 'dice2':
      from deepctr.layers.activation import Dice
      att_activation = Dice
    return melt.layers.DinAttention(activation=att_activation, weight_normalization=FLAGS.din_normalize)
  elif pooling == 'mhead':
    return melt.layers.MultiHeadAttention(num_heads=FLAGS.his_pooling_heads)
  elif not pooling:
    # by default 
    return lambda x, y, z=None: melt.layers.SumPooling()(y, z)
  else:
     return lambda x, y, z=None: melt.layers.Pooling(pooling)(y, z)
