#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2021-07-31 08:49:41.107081
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from icecream import ic

import tensorflow as tf
import melt as mt
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dot, Embedding, Flatten

from .config import *

class Word2Vec(mt.Model):
  def __init__(self, vocab_size, embedding_dim, num_ns=4):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3, 2))
    self.flatten = Flatten()


  def call(self, input):
    self.input_ = input.copy()
    # (bs, 1)
    target = input['target']
    # (bs, 5, 1)
    context = input['context']
    # ic(target.shape, context.shape)
    # (bs, 1, 128)
    word_emb = self.target_embedding(target)
    # (bs, 5, 1, 128)
    context_emb = self.context_embedding(context)
    # (bs, 5, 1, 1)
    dots = self.dots([context_emb, word_emb])
    ret = self.flatten(dots)
    # ic(word_emb.shape, context_emb.shape, dots.shape, ret.shape)
    # ic(ret, input['y'])
    return ret

  def get_loss(self):
    from w2v import loss
    loss_fn_ = loss.loss_fn
    return self.loss_wrapper(loss_fn_)
