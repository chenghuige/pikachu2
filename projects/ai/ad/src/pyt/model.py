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
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import lele
import melt
import gezi 
logging = gezi.logging

from projects.ai.ad.src.config import *

class ClsModel2(nn.Module):
  def __init__(self):
    super(ClsModel2, self).__init__() 

    Encoder = lele.layers.StackedBRNN
    # self.cemb = tf.keras.layers.Embedding(3420000, FLAGS.emb_size, name='cemb')

    self.aemb = nn.Embedding(FLAGS.vocab_size, FLAGS.emb_size)
    self.piemb = nn.Embedding(70000, FLAGS.emb_size)
    self.pemb = nn.Embedding(20, FLAGS.emb_size)
    self.iemb = nn.Embedding(400, FLAGS.emb_size)
    self.temb = nn.Embedding(100, FLAGS.emb_size)
    # self.ctemb = tf.keras.layers.Embedding(200, FLAGS.emb_size, name='ctemb')

    #isize = FLAGS.emb_size * 5
    isize = FLAGS.emb_size 
    self.encoder = Encoder(
            input_size=isize,
            hidden_size=FLAGS.hidden_size,
            num_layers=1,
            dropout_rate=0.2,
            dropout_output=False,
            recurrent_dropout=True,
            concat_layers=True,
            rnn_type='lstm',
            padding=True,
        )    

    isize = FLAGS.hidden_size
    self.pooling = lele.layers.Poolings(
                        'max,sum', 
                        input_size=isize,
                        att_activation=getattr(F, 'relu'))  

    #isize = 640
    isize = 128
    self.dense_age = nn.Linear(isize, 10)
    self.dense_gender = nn.Linear(isize, 1)

    lele.keras_init(self, True, True)

  def forward(self, input):
    # gezi.set('input', input)
    LEN = FLAGS.max_len
    x_in = input['ad_ids'][:,:LEN]
    x_mask = x_in.eq(0)
   
    x_a = self.aemb(x_in)
    #x_pi = self.piemb(input['product_ids'][:,:LEN])
    #x_p = self.pemb(input['product_categories'][:,:LEN])
    #x_i = self.iemb(input['industries'][:,:LEN])
    #x_t = self.temb(input['times'][:,:LEN])

    #x = torch.cat([x_a, x_p, x_pi, x_i, x_t], axis=-1)
    #x = torch.cat([x_a, x_p, x_i, x_t], axis=-1)
    x = x_a

    x = self.pooling(x, x_mask)

    self.age = self.dense_age(x)
    self.gender = self.dense_gender(x)

    self.pred_age = torch.argmax(self.age, axis=1)
    self.pred_gender = torch.sigmoid(self.gender) 

    return self.gender
