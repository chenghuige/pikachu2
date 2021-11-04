#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2021-01-10 21:48:28.831499
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import copy
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import gezi
from gezi import tqdm
import melt as mt
from .config import *

def get_info_keys():
  keys = FLAGS.feats + FLAGS.feats2
  if 'tag' in ''.join(keys):
    keys.append('machine_tag_probs')
  return set(keys)

def ensemble(dfs, weights=None):
  if dfs:
    for action in ACTION_LIST:
      dfs[0][action] = np.average(np.stack([dfs[i][action].values for i in range(len(dfs))], 1), 1, weights=weights)
  return dfs[0]

def mean_unk(emb):
  l2 = tf.reduce_sum(emb * emb, -1, keepdims=True)
  mask = l2 > 0
  mask = tf.cast(mask, emb.dtype)
  mean_emb = tf.reduce_mean(emb, 0, keepdims=True)
  return emb * mask + mean_emb * (1 - mask)

def mask_key(indexes, key):
  if FLAGS.mask_rate > 0.:
    indexes = aug(indexes, FLAGS.mask_rate, UNK_ID)
  else:
    mask = tf.cast(indexes > vocabs[key].size(FLAGS.min_count), indexes.dtype) 
    indexes = indexes * (1 - mask) + UNK_ID * mask
  return indexes

def aug(x, rate, unk_id=1, x_mask=None):
  if x_mask is None:
    x_mask = x > 0
  x_mask = tf.cast(x_mask, dtype=x.dtype)
  ratio = tf.random.uniform([1,], 0, rate)
  mask = tf.random.uniform([mt.get_shape(x, 0), mt.get_shape(x, 1)])  > ratio
  mask = tf.cast(mask, dtype=x.dtype)
  rmask = unk_id * (1 - mask)
  x = (x * mask + rmask) * x_mask
  return x

# turn UNK_ID(1) to 0, 这样当前是0 但是history是1 避免match匹配
def map_unk(indexes):
  mask = tf.cast(indexes != UNK_ID, tf.int32)
  ret = indexes * mask
  return ret

def need_lookup(input, feats):
  for feat in feats:
    if feat not in input:
      return True
  return False

def get_emb_kwargs():
  kwargs = {
     'mask_zero': FLAGS.mask_zero,
     'embeddings_regularizer': keras.regularizers.l2(FLAGS.embeddings_regularizer) if FLAGS.embeddings_regularizer > 0 else None
    } 
  return kwargs

v = vocabs
def pretrain_emb_kwargs(name, **kwargs):
  kwargs_ = get_emb_kwargs()
  kwargs_.update(kwargs)

  width = None

  pretrain_loaded = False
  if 'embeddings_initializer' not in kwargs_:
    npy = None
    try:
      emb_flag = getattr(FLAGS, f'{name}_emb')
    except Exception as e:
      emb_flag = None
    if emb_flag:
      day = FLAGS.pretrain_day if not FLAGS.online else FLAGS.pretrain_day_online
      if FLAGS.emb_dim != 128:
        if not emb_flag.endswith('_128'):
          emb_flag += f'_{FLAGS.emb_dim}'
        else:
          emb_flag = '_'.join(emb_flag.split('_')[:-1] + [f'{FLAGS.emb_dim}'])
      if day:
        emb_file = f'../input/{day}/{emb_flag}.npy'
        if not os.path.exists(emb_file):
          emb_file = f'../input/{emb_flag}.npy'
        if os.path.exists(emb_file):
          logging.ice('loading pretrain', name, emb_file, gezi.md5sum(emb_file))
          npy = np.load(emb_file)
          ic(npy.shape)
          width = npy.shape[1]
          # if FLAGS.num_padding_embs and name in unk_keys:
          #   npy = np.concatenate([npy,  np.random.uniform(-0.05, 0.05,(FLAGS.num_padding_embs, width))], 0)
          pretrain_loaded = True
        else:
          if FLAGS.work_mode == 'train':
            raise ValueError(emb_file)

    if npy is not None:
      kwargs_['embeddings_initializer'] = keras.initializers.constant(npy)
    elif FLAGS.embeddings_initializer == 'random_normal':
      kwargs_['embeddings_initializer'] = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)  
    elif FLAGS.embeddings_initializer == 'gnormal':
      kwargs_['embeddings_initializer'] = tf.keras.initializers.GlorotNormal()
    elif FLAGS.embeddings_initializer == 'guniform':
      kwargs_['embeddings_initializer'] = tf.keras.initializers.GlorotUniform()
    elif FLAGS.embeddings_initializer == 'hnormal':
      kwargs_['embeddings_initializer'] = tf.keras.initializers.HeNormal()
    elif FLAGS.embeddings_initializer == 'huniform':
      kwargs_['embeddings_initializer'] = tf.keras.initializers.HeUniform()
    else:
      kwargs_['embeddings_initializer'] = FLAGS.embeddings_initializer
  if 'trainable' not in kwargs_:
    if pretrain_loaded and FLAGS.embs_trainable is not None:
      kwargs_['trainable'] = FLAGS.embs_trainable
    else:
      try:
        kwargs_['trainable'] = getattr(FLAGS, f'{name}_trainable')
      except Exception:
        kwargs_['trainable'] = True
  if 'name' not in kwargs_:
    # 2021-07-19 21:24:40 0:00:32 WARNING: The name "doc" is used 2 times in the model. All layer names should be unique ...  why? doc_emb is ok
    kwargs_['name'] = f'{name}_emb'
    
  logging.ice(f'Embedding {name}', kwargs_)
  return kwargs_, width

def get_embedding(name, height=None, width=None, **kwargs):
  if height is None:
    height = v[name].size() 
    # if FLAGS.num_padding_embs and name in unk_keys:
    #   height += FLAGS.num_padding_embs
  width = width or FLAGS.emb_dim
  pargs = pretrain_emb_kwargs
  Embedding = keras.layers.Embedding if not FLAGS.use_pr_embedding else mt.layers.PrEmbedding
  kwargs_, width_ = pargs(name, **kwargs)
  width = width_ or width 
  ic(height, width)
  return Embedding(height, width, **kwargs_)

def get_doc_lookup():
  doc_lookup_file = '../input/doc_lookup.npy' 
  logging.ice('doc_lookup_file', doc_lookup_file, gezi.md5sum(doc_lookup_file))
  doc_lookup_npy = np.load(doc_lookup_file)
  # here int32 is faster
  return keras.layers.Embedding(doc_lookup_npy.shape[0], doc_lookup_npy.shape[1], 
    embeddings_initializer=tf.constant_initializer(doc_lookup_npy),
    trainable=False, dtype=tf.int32, name='doc_lookup')

lookups = {}

def init_lookups():
  for key in info_keys:
    if key not in lookups:
      lookups[key] = init_lookup(key)

def init_lookup(key):
  lookup_file = f'../input/{key}_lookup.npy'
  lookup_npy = np.load(lookup_file)
  logging.ice('lookup_file', lookup_file, lookup_npy.shape, gezi.md5sum(lookup_file))
  return keras.layers.Embedding(lookup_npy.shape[0], lookup_npy.shape[1], 
    embeddings_initializer=tf.constant_initializer(lookup_npy),
    trainable=False, dtype=tf.int32, name=f'{key}_lookup')

def lookup(key, indexes):
  return lookups[key](indexes)

def get_encoder(encoder):
  if encoder is None:
    encoder = lambda x, y: x
  elif encoder.upper() in ['LSTM', 'GRU']:
    # encoder = mt.layers.CudnnRnn(num_layers=FLAGS.num_layers, 
    #                             num_units=int(FLAGS.emb_dim / 2), 
    #                             keep_prob=1. - FLAGS.dropout,
    #                             share_dropout=False,
    #                             recurrent_dropout=False,
    #                             concat_layers=FLAGS.concat_layers,
    #                             bw_dropout=False,
    #                             residual_connect=False,
    #                             train_init_state=False,
    #                             cell=encoder)
    ## TODO length ?
    RNN = getattr(tf.keras.layers, encoder.upper())
    if FLAGS.rnn_strategy == 'bi':
      encoder_ = tf.keras.layers.Bidirectional(RNN(int(FLAGS.emb_dim / 2), return_sequences=True))
      # encoder = lambda x, y: encoder_(x)
      encoder = lambda x, y: encoder_(x, mask=tf.sequence_mask(y, mt.get_shape(x, 1)))
    else:
      encoder_ = RNN(FLAGS.emb_dim , return_sequences=True) 
      if FLAGS.rnn_strategy == 'forward':
        encoder = lambda x, y: encoder_(x, mask=tf.sequence_mask(y, mt.get_shape(x, 1)))
      else:
        encoder = lambda x, y: tf.reverse_sequence(encoder_(tf.reverse_sequence(x, y, seq_axis=1), 
                                                              mask=tf.sequence_mask(y, mt.get_shape(x, 1))),
                                                   y, seq_axis=1)

  elif encoder == 'autoint':
    encoder = mt.layers.Autoint(FLAGS.emb_dim, FLAGS.num_heads, FLAGS.num_layers)
  elif encoder == 'cross':
    encoder = mt.layers.Cross(FLAGS.cross_layers)
  elif encoder == 'self_att':
    encoder = mt.layers.SelfAttnMatch(FLAGS.dropout)
  elif encoder == 'mhead_self_att' or encoder == 'mhead':
    encoder = mt.layers.MultiHeadSelfAttention(num_heads=FLAGS.num_heads)
  elif encoder == 'transformer':
    # TODO ValueError: Weights for model sequential_6 have not yet been created. Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.
    encoder = mt.layers.transformer.Encoder(FLAGS.num_layers, FLAGS.num_heads, FLAGS.emb_dim, FLAGS.emb_dim)

    # from deepctr.layers.sequence import Transformer, AttentionSequencePoolingLayer
    # encoder = Transformer()
  elif encoder == 'nrms': #TODO 
    # ValueError: Dimension must be 5 but is 4 for '{{node model/self_attention/transpose_7}} = Transpose[T=DT_FLOAT, Tperm=DT_INT32](model/self_attention/truediv, model/self_attention/transpose_7/perm)' with input shapes: [2048,4,50,4,50], [4].
    from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention
    encoder = lambda x, y: SelfAttention(FLAGS.num_heads, FLAGS.emb_dim, seed=0)([x] * 3)
    # encoder = lambda x, y: SelfAttention(FLAGS.num_heads, int(FLAGS.emb_dim / FLAGS.num_heads), seed=0)([x] * 3)
  else:
    raise ValueError(encoder)
  return encoder

# def get_att_pooling(pooling, name=None):
#   if pooling == 'din':
#     # att_activation = FLAGS.att_activation
#     att_activation = 'sigmoid'
#     if att_activation == 'dice2':
#       from deepctr.layers.activation import Dice
#       att_activation = Dice
#     return mt.layers.DinAttention(activation=att_activation, weight_normalization=True, name=name)
#   elif pooling == 'mhead':
#     return mt.layers.MultiHeadAttention(num_heads=FLAGS.num_heads, name=name)
#   elif not pooling:
#     # by default 
#     # return lambda x, y, z:  mt.layers.SumPooling()(y, z)
#     return mt.layers.PoolingWrapper('sum', name=name)
#   else:
#     #  return lambda x, y, z: mt.layers.Pooling(pooling, name=name)(y, z)
#     return mt.layers.PoolingWrapper(pooling, name=name)
