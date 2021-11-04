#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-26 23:00:24.215922
#   \Description  
# ==============================================================================
"""
depreciated just use tfrecord_dataset.py 
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

import melt 
logging = melt.logging
import numpy as np

from projects.feed.rank.src.config import *

class Dataset(melt.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)
    self.Type = tf.data.TextLineDataset
    self.batch_parse = FLAGS.batch_parse
    self.index_addone = int(FLAGS.index_addone)
    assert self.index_addone
    self.max_feat_len = FLAGS.max_feat_len

    self.field_id = {} # field name -> field id
    self.feat_to_field = {} # feat id -> field id
    self.feat_to_field_val = {} # feat id -> field value
    self.feat_to_field_name = {}
    if not FLAGS.hash_encoding:
      self.load_feature_files()
    self.batch_size = melt.batch_size() 
    #---np.float32 much slower.. 1.0 -> 1.5h per epoch..
    self.float_fn = float if self.batch_parse else np.float32

    # feature idx start from 4
    self.start = 4

  def load_feature_files(self):
    for i, line in enumerate(open(FLAGS.feat_file_path, 'r')):
      if line == '':
        break
      line = line.rstrip()
      fields = line.split('\t')
      if len(fields) == 2:
        fid = int(fields[1]) 
      else:
        fid = i + 1

      tokens = fields[0].split('\a')
      if tokens[0] not in self.field_id:
        self.field_id[tokens[0]] = len(self.field_id)  + 1
      self.feat_to_field[fid] = self.field_id[tokens[0]]
      self.feat_to_field_val[fid] = tokens[1]
      self.feat_to_field_name[fid] = tokens[0]
    with open(FLAGS.field_file_path, 'w') as out:
      l = sorted(self.field_id.items(), key = lambda x: x[1])
      for filed, fid in l:
        print(filed, fid, sep='\t', file=out)

    logging.info('----num fields', len(self.field_id))

    # self.doc_emb_field_id = -1
    # self.user_emb_field_id = -1

    # if FLAGS.doc_emb_name in self.field_id:
    #   self.doc_emb_field_id = self.field_id[FLAGS.doc_emb_name]
    # if FLAGS.user_emb_name in self.field_id:
    #   self.user_emb_field_id = self.field_id[FLAGS.user_emb_name]

    # logging.info('----doc_emb_field_id', self.doc_emb_field_id)
    # logging.info('----user_emb_field_id', self.user_emb_field_id)

  def get_feat(self, fields):
    num_features = len(fields) 
    feat_id = [None] * num_features
    feat_field = [None] * num_features
    feat_value = [None] * num_features

    for i in range(num_features):
      tokens = fields[i].split(':')
      feat_id[i] = int(tokens[0])
      feat_field[i] = self.feat_to_field[feat_id[i]]
      feat_value[i] = self.float_fn(tokens[1])

    return feat_id, feat_field, feat_value

  # 添加文本emb， 用户emb 雅琼特征 TODO 目前只添加了doc emb 另外应该采用类似用户画像的特征加入方式 
  def get_feat_emb(self, fields):
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

  # 添加用户画像 捷鑫特征
  def get_feat_portrait(self, fields):
    has_portrait = True
    if ',' in fields[-1]:
      num_features = len(fields) - 3
    else:
      has_portrait = False
      num_features = len(fields)
    feat_id = [None] * num_features
    feat_field = [None] * num_features
    feat_value = [None] * num_features

    for i in range(num_features):
      tokens = fields[i].split(':')
      feat_id[i] = int(tokens[0])
      feat_field[i] = self.feat_to_field[feat_id[i]]
      feat_value[i] = self.float_fn(tokens[1])
    
    if has_portrait:
      cycle_profile_click = list(map(float, fields[-3].split(':')[1].split(',')))
      cycle_profile_show = list(map(float, fields[-2].split(':')[1].split(',')))
      cycle_profile_dur = list(map(float, fields[-1].split(':')[1].split(',')))
    else:
      cycle_profile_click = [0.] * FLAGS.portrait_emb_dim
      cycle_profile_show = [0.] * FLAGS.portrait_emb_dim
      cycle_profile_dur = [0.] * FLAGS.portrait_emb_dim
      
    assert len(cycle_profile_click) == FLAGS.portrait_emb_dim, fields
    assert len(cycle_profile_show) == FLAGS.portrait_emb_dim, fields
    assert len(cycle_profile_dur) == FLAGS.portrait_emb_dim, fields

    return feat_id, feat_field, feat_value, cycle_profile_click, cycle_profile_show, cycle_profile_dur

  #-----------------------depreciated below, juse use above for using Tfrecord
  
  #-----------by this way decode line by line , more powerfull, but slower if batch parse then you must have fixed batch size! 1epoch:[2.69h] batch parse 2.03h
  # batch parse means the final batch is also batch_size not smaller, so it will contains empty examples like id=='', see read-test3.py
  def parse_line(self, line, decode=True):
    # tf will convert as bytes... so need decode at first
    if decode:
      line = line.decode()
    fields = line.split('\t')
    #need np.float32 if float32 tf complain double .., but np.lofat32 is much slower then float
    label = np.float32(fields[0])
    id = '{}\t{}'.format(fields[2], fields[3])
    feat_id, feat_field, feat_value = self.get_feat(fields[self.start:])
    # need [label] consider tfrecord generation
    return feat_id, feat_field, feat_value, [label], [id]


  def line_parse_(self, line):
    feat_id, feat_field, feat_value, label, id = \
        tf.compat.v1.py_func(self.parse_line, [line],
                    [tf.int64, tf.int64, tf.float32, tf.float32, tf.string])
    feat_id.set_shape([None])
    feat_field.set_shape([None])
    feat_value.set_shape([None])
    label.set_shape([1]) 
    id.set_shape([1])
    # label id shape like (batch_size,)
    return {'index': feat_id, 'field': feat_field, 'value': feat_value, 'id': tf.squeeze(id, -1)}, tf.squeeze(label, -1)

  # https://stackoverflow.com/questions/52284951/tensorflow-py-func-typeerror-with-tf-data-dataset-output
  def parse_batch(self, feat_list, batch_size):
    feat_ids = np.zeros((batch_size, self.max_feat_len), dtype=np.int64)
    feat_fields = np.zeros((batch_size, self.max_feat_len), dtype=np.int64)
    feat_values = np.zeros((batch_size, self.max_feat_len), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.float32)
    ids = [''] * batch_size # ''means not effective id, usefull for batch_parse + not repeat final batch with padding elments

    # doc_embs = [None] * batch_size
    # user_embs = [None] * batch_size

    cur_max_feat_len = 0
    for i, feat_line in enumerate(feat_list):
      # python 3 need decode
      fields = feat_line.decode().split('\t')
      assert len(fields) > self.start, fields
      #fields = feat_line.split('\t')
      labels[i] = float(fields[0])
      ids[i] = '{}\t{}'.format(fields[2], fields[3])

      # feat_id, feat_field, feat_value, doc_emb, user_emb = self.get_feat(fields[self.start:])
      feat_id, feat_field, feat_value = self.get_feat(fields[self.start:])

      #assert len(feat_id) == len(feat_value), "len(feat_id) == len(feat_value) -----------------"
      trunc_len = min(len(feat_id), self.max_feat_len)
      #---也许是因为批量写速度比逐个访问numpy数组位置快(原地逐个访问)
      feat_ids[i, :trunc_len] = feat_id[:trunc_len]
      feat_fields[i, :trunc_len] = feat_field[:trunc_len]
      feat_values[i, :trunc_len] = feat_value[:trunc_len]
      cur_max_feat_len = max(cur_max_feat_len, trunc_len)
      # doc_embs[i] = doc_emb
      # user_embs[i] = user_emb

    ## even here [:i, :cur..] still final batch size is same not small 
    # feat_ids = feat_ids[:i, :cur_max_feat_len]
    # feat_fields = feat_fields[:i, :cur_max_feat_len]
    # feat_values = feat_values[:i, :cur_max_feat_len]
    feat_ids = feat_ids[:, :cur_max_feat_len]
    feat_fields = feat_fields[:, :cur_max_feat_len]
    feat_values = feat_values[:, :cur_max_feat_len]
    # doc_embs = np.array(doc_embs)
    # user_embs = np.array(user_embs)

    labels = labels.reshape(-1, 1)
    # return feat_ids, feat_fields, feat_values, doc_embs, user_embs, labels, ids
    return feat_ids, feat_fields, feat_values, labels, ids


  def batch_parse_(self, line, batch_size):
    doc_embs = None
    user_embs = None

    # feat_ids, feat_fields, feat_values, doc_embs, user_embs, labels, ids = \
    #     tf.compat.v1.py_func(self.parse_batch, [line, batch_size],
    #                 [tf.int64, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.string])
    feat_ids, feat_fields, feat_values, labels, ids = \
        tf.compat.v1.py_func(self.parse_batch, [line, batch_size],
                    [tf.int64, tf.int64, tf.float32, tf.float32, tf.string])
    
    #---for pyfunc you need to set shape.. otherwise first dim unk strange for keras layer TODO FIXME
    feat_ids.set_shape((batch_size, None))
    feat_fields.set_shape((batch_size, None))
    feat_values.set_shape((batch_size, None))

    if doc_embs is not None:
      doc_embs.set_shape((batch_size, FLAGS.doc_emb_dim))
    if user_embs is not None:
      user_embs.set_shape((batch_size, FLAGS.user_emb_dim))

    #return {'index': feat_ids, 'field': feat_fields, 'value': feat_values, 'id': ids}, labels
    X = {'index': feat_ids, 'field': feat_fields, 'value': feat_values, 'id': ids} 

    if doc_embs is not None:
      X['doc_emb'] = doc_embs

    if user_embs is not None:
      X['user_emb'] = user_embs

    y = labels
    return X, y

  def parse(self, line, batch_size):
    if self.batch_parse:
      return self.batch_parse_(line, batch_size)
    else:
      return self.line_parse_(line)

