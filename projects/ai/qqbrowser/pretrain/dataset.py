#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2021-07-31 08:49:52.078016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from pandas.io import feather_format
from icecream import ic
import tensorflow as tf

from gezi import tqdm
from .config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='train', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    keys, excl_keys = ['title_ids', 'title_mask', 'title_words_mask', 
                      'title_word_ids', 'input_ids', 'word_ids', 
                      'attention_mask', 'token_type_ids'], []
    if FLAGS.use_vision:
      keys += ['frames', 'num_frames']
    self.auto_parse(keys=keys, exclude_keys=excl_keys)
    fe = self.parse_(serialized=example)

    mt.try_append_dim(fe)
    
    x = {}
    if not 'input_ids' in fe:
      x['input_ids'] = fe['title_ids'] if not FLAGS.word else  fe['title_word_ids']
      # x['attention_mask'] = fe['title_mask']
      # x['title_words_mask'] = fe['title_words_mask']
    else:
      if FLAGS.word:
        x['input_ids'] = fe['word_ids']
      else:
        x['input_ids'] = fe['input_ids']
      
    # x['attention_mask'] = tf.cast(x['input_ids'] > 0, x['input_ids'].dtype)
    # else:
    #   for key in keys:
    #     if key in fe:
    #       x[key] = fe[key][:,:32]
    if FLAGS.vocab_size:
      mask = tf.cast(x['input_ids'] < FLAGS.vocab_size, x['input_ids'].dtype)
      x['input_ids'] = x['input_ids'] * mask + tf.ones_like(mask) * (1 - mask)

    if FLAGS.word:
      x['input_ids'] = x['input_ids'][:,:64]

    if FLAGS.use_vision:
      x['embs'] = tf.reshape(fe['frames'], [-1, FLAGS.max_frames, FLAGS.frame_embedding_size])
      x['embs_attention_mask'] = tf.sequence_mask(fe['num_frames'], FLAGS.max_frames)

    y = x['input_ids']
    return x, y
