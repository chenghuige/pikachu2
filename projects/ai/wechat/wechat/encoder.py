#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   encoder.py
#        \author   chenghuige  
#          \date   2021-01-10 21:48:28.831499
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import copy
from icecream import ic

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import gezi
from gezi import tqdm
import melt as mt
from .config import *
from .util import *

# 这里encoder输入是 (bs, seqs) 经过emb (bs, seqs, emb_dim) 输出是 (bs, emb_dim) 相当于最后做pooling压缩第一维度
class Encoder(keras.Model):
  def __init__(self, emb=None, encoder=None, pooling=None, emb_dim=None, split=False, num_lookups=None, float_base=1e7, return_sequences=False,**kwargs):
    super(Encoder, self).__init__(**kwargs)   
    self.emb = emb
    pooling = pooling or 'att'
    if isinstance(pooling, str):
      self.pooling = mt.layers.Pooling(pooling)
    else:
      self.pooling = pooling

    self.output_dim = None
    if emb is not None:
      self.output_dim = emb.output_dim
    self.encoder = get_encoder(encoder) if isinstance(encoder, str) or encoder is None else encoder 

    if not self.output_dim:
      try:
        self.output_dim = self.pooling.output_dim
      except Exception:
        self.output_dim = emb_dim

    self.emb_dim = emb_dim or self.emb.output_dim

    self.split = split
    self.num_lookups = num_lookups
    self.float_base = float_base

    self.return_sequences = return_sequences

  # in eger mode ok without below but graph not
  def compute_output_shape(self, input_shape):
    # input_shape: TensorShape([4096, 4, 128])
    # ic(input_shape) 
    out_shape = input_shape[:1].concatenate(self.output_dim)
    # out_shape: TensorShape([4096, 128])
    # ic(out_shape)
    return out_shape

  def call(self, x, weights=None, segment_ids=None):
    # notice bs -> bs * dim[1]
    # ic(x.shape) 
    if self.split or (self.num_lookups and x.shape[-1] == self.num_lookups * 2):
      x, weights = tf.split(x, 2, axis=-1)
      weights = tf.cast(weights, self.emb.dtype) / tf.cast(self.float_base, self.emb.dtype)
      
    len_ = None
    if self.emb is not None:
      embs = self.emb(x)
      len_ = mt.length(x)
      len_ = tf.math.maximum(len_, 1)
    else:
      embs = x

    if weights is not None:
      weights = tf.expand_dims(weights, -1)
      embs *= tf.cast(weights, embs.dtype)

    if segment_ids is None:
      # ic(embs.shape)
      xs = self.encoder(embs, len_)
      # ic(xs.shape)
    else:
      xs = self.encoder(embs, segment_ids)

    if self.return_sequences:
      return xs
    
    # ic(xs.shape)
    xs = self.pooling(xs, len_)
    # ic(xs.shape)
    return xs

class SeqsEncoder(keras.Model):
  def __init__(self, encoder, seqs_encoder=None, pooling=None, use_seq_position=False, return_sequences=False, **kwargs):
    super(SeqsEncoder, self).__init__(**kwargs)   
    self.encoder_ = encoder
    self.encoder = keras.layers.TimeDistributed(encoder, name=encoder.name)
    self.seqs_encoder = get_encoder(seqs_encoder)
    pooling = pooling or 'att'
    if isinstance(pooling, str):
      self.pooling = mt.layers.Pooling(pooling)
    else:
      self.pooling = pooling
    # position emb seems not help 
    # self.use_seq_position = use_seq_position or FLAGS.use_seq_position
    self.use_seq_position = False
    if self.use_seq_position:
      self.position_emb = keras.layers.Embedding(500, FLAGS.emb_size)
    self.return_sequences = return_sequences 

  def call(self, seqs, tlen, query=None, segment_ids=None, return_sequences=False):
    # ic(seqs.shape)
    input_shape = seqs.shape
    seqs = self.encoder(seqs)
    # self.encoder.compute_output_shape(input_shape): TensorShape([4096, 50, 128])
    # ic(self.encoder.compute_output_shape(input_shape))
    # ic(seqs.shape)
    if segment_ids is None:
      seqs = self.seqs_encoder(seqs, tlen)
    else:
      seqs = self.seqs_encoder(seqs, segment_ids)
    if self.use_seq_position:
      bs = tf.shape(seqs)[0]
      max_len = tf.shape(seqs)[1]
      positions = tf.tile(tf.expand_dims(tf.range(max_len), 0),[bs, 1])
      position_embs = self.position_emb(positions)
      if 'din' in self.pooling:
        return self.pooling(query, seqs, tlen, context=position_embs)
      else:
        seqs += position_embs

    if return_sequences or self.return_sequences:
      return seqs

    res = self.pooling(seqs, tlen, query)
    return res

class Encoders(keras.Model):
  def __init__(self, embs, encoder=None, pooling=None, combiner='sum', **kwargs):
    super(Encoders, self).__init__(**kwargs)   
    self.embs = embs
    self.encoder = get_encoder(encoder)
    pooling = pooling or 'att'
    self.pooling = mt.layers.Pooling(pooling)
    self.output_dim = embs[0].output_dim if combiner != 'concat' else embs[0].output_dim * (len(embs))
    self.combiner = combiner
    assert self.combiner == 'sum'

  # in eger mode ok without below but graph not
  def compute_output_shape(self, input_shape):
    return (None, self.output_dim)

  def call(self, x, segment_ids=None):
    xs = tf.split(x, len(self.embs), axis=-1)
    embs = []
    for x, emb in zip(xs, self.embs):
      embs += [emb(x)]

    embs = tf.add_n(embs)
    len_ = mt.length(x)
    if segment_ids is None:
      seqs = self.encoder(embs, len_)
    else:
      seqs = self.encoder(embs, segment_ids)
    return self.pooling(seqs, len_)

class HisEncoder(keras.Model):
  def __init__(self, encoder, **kwargs):
    super(HisEncoder, self).__init__(**kwargs)   
    self.encoder = get_encoder(encoder) if isinstance(encoder, str) or encoder is None else encoder

  def call(self, x, len_, query=None):
    if query is None:
      res = self.encoder(x, len_)
      return res
    else:
      if len(query.shape) < len(x.shape):
        query = tf.expand_dims(query, 1)
      x = tf.concat([query, x], 1)
      x = self.encoder(x, len_ + 1)
      res, query = x[:,1:], x[:,0]
      return res, query