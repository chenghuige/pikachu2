#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-26 23:00:24.215922
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt 
logging = melt.logging
import numpy as np

from config import *

class Dataset(melt.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)
    self.Type = tf.data.TextLineDataset
    self.batch_parse = FLAGS.batch_parse
    self.index_addone = int(FLAGS.index_addone)
    assert self.index_addone
    self.max_feat_len = FLAGS.max_feat_len

    self.field_id = {}
    self.feat_to_field = {}
    self.feat_to_field_val = {}
    self.load_feature_files()
    self.batch_size = melt.batch_size() 
    #---np.float32 much slower.. 1.0 -> 1.5h per epoch..
    self.float_fn = float if self.batch_parse else np.float32

    # feature idx start from 4
    self.start = 4

  def load_feature_files(self):
    for line in open(FLAGS.feat_file_path, 'r'):
      if line == '':
        break
      line = line.rstrip()
      fields = line.split('\t')
      assert len(fields) == 2
      fid = int(fields[1]) 

      tokens = fields[0].split('\a')
      if tokens[0] not in self.field_id:
        self.field_id[tokens[0]] = len(self.field_id)  + 1
      self.feat_to_field[fid] = self.field_id[tokens[0]]
      self.feat_to_field_val[fid] = tokens[1]
    with open(FLAGS.field_file_path, 'w') as out:
      l = sorted(self.field_id.items(), key = lambda x: x[1])
      for filed, fid in l:
        print(filed, fid, sep='\t', file=out)

    if FLAGS.doc_emb_name in self.field_id:
      self.doc_emb_field_id = self.field_id[FLAGS.doc_emb_name]
    if FLAGS.user_emb_name in self.field_id:
      self.user_emb_field_id = self.field_id[FLAGS.user_emb_name]

    logging.info('----num fields', len(self.field_id))
    logging.info('----doc_emb_field_id', self.doc_emb_field_id)
    logging.info('----user_emb_field_id', self.user_emb_field_id)


  def get_feat(self, fields):
    doc_emb = [0.] * FLAGS.doc_emb_dim
    user_emb = [0.] * FLAGS.user_emb_dim 
    for j in reversed(range(len(fields))):
      tokens = fields[j].split(':')
      feat_id = int(tokens[0])
      if feat_id < FLAGS.emb_start:
        break
      else:
        if self.feat_to_field[feat_id] == self.doc_emb_field_id:
          doc_emb[int(self.feat_to_field_val[feat_id])] = self.float_fn(tokens[1])
    j += 1
    #num_features = len(fields) 
    num_features = j
    feat_id = [None] * num_features
    feat_field = [None] * num_features
    feat_value = [None] * num_features

    for i in range(num_features):
      tokens = fields[i].split(':')
      feat_id[i] = int(tokens[0])
      feat_field[i] = self.feat_to_field[feat_id[i]]
      feat_value[i] = self.float_fn(tokens[1])

    return feat_id, feat_field, feat_value, doc_emb, user_emb
  
