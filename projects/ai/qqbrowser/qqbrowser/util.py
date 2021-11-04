#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2021-08-26 04:09:58.773709
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

import gezi
from .config import *

def is_pointwise():
  return FLAGS.parse_strategy < 3

def is_pairwise():
  return not is_pointwise()
  
def get_bert_embedding(out, pooling_strategy='cls'):
  if pooling_strategy == 'pooler':
    if 'pooler_output' in out:
      return out['pooler_output']
    else:
      return out[0][:, 0]
  
  out = out[0]
  
  if pooling_strategy in ['cls', 'first']:
    return out[:, 0]
  
  if pooling_strategy in ['avg', 'mean']:
    return tf.reduce_mean(out, axis=1)
  
  if pooling_strategy in ['sum']:
    return tf.reduce_sum(out, axis=1)
  
  if pooling_strategy in ['max']:
    return tf.reduce_max(out, axis=1)
  
  return out[:, 0]

def get_datasets():
  from baseline.tensorflow.config import parser
  from baseline.tensorflow.data_helper import FeatureParser
  
  args = parser.parse_args([])
  fparser = FeatureParser(args)

  train_pattern = f'../input/pointwise/*.tfrecords'
  ic(train_pattern)
  train_files=gezi.list_files(train_pattern)
  ic(len(train_files), train_files[:5])
  # ic(FLAGS.batch_size, mt.batch_size(), FLAGS.eval_batch_size, mt.eval_batch_size()) # 注意mt.init之后修改都变成了 batch_size_per_gpu, global batch size通过mt.batch_size()访问
  train_dataset = fparser.create_dataset(train_files, training=True, batch_size=mt.batch_size(), return_labels=True, repeat=True)
  if not FLAGS.num_train:
    FLAGS.num_train = gezi.read_int('../input/num_train.txt')
    if not FLAGS.num_train:
      FLAGS.num_train = mt.get_num_records(train_files, recount=True)
      gezi.write_txt(FLAGS.num_train, '../input/num_train.txt')
  ic(FLAGS.num_train)

  val_pattern = f'../input/pairwise/*.tfrecords'
  ic(val_pattern)
  val_files = gezi.list_files(val_pattern)
  ic(len(val_files), val_files[:5])
  eval_dataset = fparser.create_dataset(val_files, training=False, batch_size=mt.eval_batch_size(), return_labels=True)
  val_dataset = fparser.create_dataset(val_files, training=False, batch_size=mt.eval_batch_size(), return_labels=True, repeat=True)
  if not FLAGS.num_valid:
    FLAGS.num_valid = gezi.read_int('../input/num_valid.txt')
    if not FLAGS.num_valid:
      FLAGS.num_valid = mt.get_num_records(val_files, recount=True)
      gezi.write_txt(FLAGS.num_valid, '../input/num_valid.txt')
  ic(FLAGS.num_valid)

  test_part = 'a'
  test_pattern = f'../input/test_{test_part}/*.tfrecords'
  ic(test_pattern)
  test_files = gezi.list_files(test_pattern)
  ic(len(test_files), test_files[:5])
  test_dataset = fparser.create_dataset(test_files, training=False, batch_size=mt.eval_batch_size())
  FLAGS.num_test = gezi.read_int('../input/num_test.txt')
  if not FLAGS.num_test:
    FLAGS.num_test = mt.get_num_records(test_files, recount=True)
    gezi.write_txt(FLAGS.num_test, '../input/num_test.txt')
  ic(FLAGS.num_test)
  
  return train_dataset, (eval_dataset, val_dataset), test_dataset

def split_pairwise(inputs):
  pairwise = {
    1: {},
    2: {}
  }
  for key in inputs:
    if key.endswith('1') or key.endswith('2'):
      index = int(key[-1])
      key_ = key[:-1]
      pairwise[index][key_] = inputs[key]
    else:
      pairwise[1][key] = inputs[key]
      pairwise[2][key] = inputs[key]
  return pairwise[1], pairwise[2]
