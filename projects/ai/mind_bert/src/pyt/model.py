#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2020-04-28 17:49:25.853292
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

import gezi
logging = gezi.logging
import lele

from transformers import AutoModel

class XlmModel(nn.Module):
  def __init__(self):
    super(XlmModel, self).__init__() 

    pretrained = FLAGS.pretrained

    with gezi.Timer(f'Torch load xlm_model from {pretrained}', True, logging.info):
      self.transformer = AutoModel.from_pretrained(pretrained, from_tf=True)

    idim = 768
    odim = len(toxic_types) + 1 if FLAGS.multi_head else 1

    if not FLAGS.use_multi_dropout:
      self.dense = nn.Sequential(nn.Linear(idim, odim), nn.Sigmoid())
    else:
      self.num_experts = 5
      self.denses1 = [nn.Sequential(nn.Linear(idim, 32), nn.Relu())] * self.num_experts
      self.denses2 = [nn.Sequential(nn.Linear(idim, 32), nn.Sigmoid())] * self.num_experts
      self.dropouts = [nn.Dropout(FLAGS.dropout)] * self.num_experts

  def forward(self, input):
    input_word_ids = input['input_word_ids']
    x = self.transformer(input_word_ids)[0]
    x = x[:, 0, :]

    # mmoe ?
    if FLAGS.use_word_ids2:
      x1 = x
      input_word_ids2 = input['input_word_ids2']
      x2 = self.transformer(input_word_ids2)[0]
      x2 = x2[:, 0, :]
      x = torch.cat([x, x2], -1)

    if not FLAGS.use_multi_dropout:
      x = self.dense(x)
    else:
      xs = []
      for i in range(self.num_experts):
        x_i = self.dropouts[i](x)
        x_i = self.denses1[i](x_i)
        x_i = self.denses2[i](x_i)
        xs += [x_i]
        
      x = tf.reduce_mean(tf.concat(xs, axis=1), 1, keepdims=True)

    # tf.print(x)
    return x

xlm_model = XlmModel