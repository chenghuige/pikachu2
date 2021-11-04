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
from projects.ai.mango.src.config import *

def out_hook(model): 
  return dict(prob=model.prob, index=model.index, did=model.did, 
              vid=model.vid, watches=model.watches)

def infer_write(ids, predicts, ofile, others):
  index = others['index']
  prob = others['prob']
  df = pd.DataFrame({'index': index, 'score': prob})
  df = df.sort_values('index')
  df.to_csv(ofile, index=False)

def valid_write(ids, labels, predicts, ofile, others):
  index = others['index']
  prob = others['prob']
  did = gezi.decode(others['did'])
  vid = others['vid']
  watches = others['watches']
  df = pd.DataFrame({'index': index, 'label': labels, 'score': prob, 'did': did, 'vid': vid, 'watches': watches})
  df = df.sort_values('index')
  df.to_csv(ofile, index=False)

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
def get_vid(vid, max_id):
  if not max_id:
    return vid
  small_vid_mask = tf.cast(vid <= max_id, tf.int64)
  large_vid_mask = 1 - small_vid_mask
  final_vid = vid * small_vid_mask + large_vid_mask
  return final_vid
 
def unk_aug(x, x_mask=None):
  """
  randomly make 10% words as unk
  TODO this works, but should this be rmoved and put it to Dataset so can share for both pyt and tf
  """
  # if not self.training or not FLAGS.unk_aug or melt.epoch() < FLAGS.unk_aug_start_epoch:
  #   return x 
  if not K.learning_phase() or not FLAGS.unk_aug_rate:
    return x
    
  if x_mask is None:
    x_mask = x > 0
  x_mask = tf.cast(x_mask, dtype=tf.int64)
  ratio = tf.random.uniform([1,], 0, FLAGS.unk_aug_rate)
  mask = tf.random.uniform([melt.get_shape(x, 0), melt.get_shape(x, 1)])  > ratio
  mask = tf.cast(mask, dtype=tf.int64)
  unk_id = 1
  rmask = unk_id * (1 - mask)
  x = (x * mask + rmask) * x_mask
  return x

def rand_aug(x, x_mask=None):
  """
  randomly make 10% words as unk
  TODO this works, but should this be rmoved and put it to Dataset so can share for both pyt and tf
  """
  # if not self.training or not FLAGS.unk_aug or melt.epoch() < FLAGS.unk_aug_start_epoch:
  #   return x 
  if not K.learning_phase() or not FLAGS.rand_aug_rate:
    return x
    
  if x_mask is None:
    x_mask = x > 0
  x_mask = tf.cast(x_mask, dtype=tf.int64)
  ratio = tf.random.uniform([1,], 0, FLAGS.rand_aug_rate)
  mask = tf.random.uniform([melt.get_shape(x, 0), melt.get_shape(x, 1)])  > ratio
  mask = tf.cast(mask, dtype=tf.int64)
  unk_id = 1
  rmask = unk_id * (1 - mask)
  x = (x * mask + rmask) * x_mask
  return x

def create_emb(vocab_name, emb_name=None):
  if emb_name is None:
    emb_name = vocab_name
  Embedding = melt.layers.VEmbedding if FLAGS.use_vocab_emb else keras.layers.Embedding
  vs = gezi.get('vocab_sizes')
  embeddings_initializer = 'uniform'
  trainable = True
  if FLAGS.use_w2v and vocab_name == 'vid':
    emb = np.load(FLAGS.vid_pretrain)
    emb = emb[:vs['vid'][0]]
    embeddings_initializer=tf.constant_initializer(emb)
    trainable = FLAGS.train_vid_emb
  if FLAGS.words_w2v and vocab_name == 'words':
    emb = np.load(FLAGS.words_pretrain)
    emb = emb[:vs['words'][0]]
    embeddings_initializer=tf.constant_initializer(emb)
    trainable = FLAGS.train_word_emb
  if FLAGS.stars_w2v and vocab_name == 'stars':
    emb = np.load(FLAGS.stars_pretrain)
    emb = emb[:vs['stars'][0]]
    embeddings_initializer=tf.constant_initializer(emb)
    trainable = FLAGS.train_stars_emb
  if vocab_name == 'image':
    emb = np.load('../input/all/image_emb.npy')
    embeddings_initializer=tf.constant_initializer(emb)
    trainable = FLAGS.train_image_emb
  logging.info(vocab_name, vs[vocab_name][0], vs[vocab_name][1], embeddings_initializer, trainable)
  if FLAGS.use_vocab_emb:
    return Embedding(vs[vocab_name][0], FLAGS.emb_size, vs[vocab_name][1], 
                    embeddings_initializer=embeddings_initializer, 
                    trainable=trainable, 
                    bias_emb=FLAGS.use_bias_emb,
                    scale_emb=FLAGS.use_scale_emb,
                    name=f'{emb_name}_emb')
  else:
    return Embedding(vs[vocab_name][0], FLAGS.emb_size, 
                embeddings_initializer=embeddings_initializer, 
                trainable=trainable, 
                name=f'{emb_name}_emb')

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
