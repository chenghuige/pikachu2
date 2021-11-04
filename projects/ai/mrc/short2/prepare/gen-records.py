#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2019-07-27 22:33:36.314010
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('..')
sys.path.append('/home/featurize/work/other/mrc-toolkit-short')
import os

from absl import app, flags
FLAGS = flags.FLAGS

import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count

from sogou_mrc.dataset.short_qa import NewDataReader
from sogou_mrc.libraries.bert_wrapper_for_sqa import BertDataHelperForSQA

import gezi
from gezi import pad
import melt

import tensorflow as tf

eval_data = None
bert_helper = None

class Merger(object):
  def __init__(self, writer):
    self.writer = writer
    self.feats = []
    self.qid = None
    self.m = {}

  def _merge(self, key, elems=1):
    N = 10
    m = self.m
    m[key] = [x[key] for x in self.feats]
    if isinstance(m[key][0], (list, tuple)):
      m[key] = list(np.concatenate(m[key]))
    m[key] = gezi.pad(m[key], N * elems)

  def merge(self):
    if self.feats:
      m = self.m
      m['qid'] = self.qid
      m['num_passages'] = len(self.feats)
      self._merge('passage_has_answer')
      self._merge('answer_start')
      self._merge('answer_end')
      self._merge('start_position')
      self._merge('end_position')
      self._merge('input_ids', 300)
      self._merge('segment_ids', 300)
      self._merge('input_mask', 300)
      self._merge('passage_word_mask', 300)
      # print(m)
      self.writer.write_feature(m)

  def __call__(self, feature):
    qid = feature['qid']
    if qid != self.qid:
      self.merge()
      self.feats = []
      self.qid = qid
    self.feats.append(feature)

  def finish(self):
    sef.merge()
    self.writer.close()

def get_fold(total, num_folds, index):
  # index is None means all
  if index is None:
    return 0, total
  elif index < 0:
    return 0, 0
  assert num_folds
  fold_size = -(-total // num_folds)
  start = fold_size * index
  end = start + fold_size if index != num_folds - 1 else total

  if index > 0:
    pre_qid = eval_data[start - 1]['qid']
    start_ = start
    while eval_data[start_]['qid'] == pre_qid:
      start_ += 1
    start = start_

  if index < num_folds - 1:
    end_ = end + 1
    pre_qid = eval_data[end]['qid']
    while eval_data[end_]['qid'] == pre_qid:
      end_ += 1
    end = end_ - 1

  return start, end

def build_features(index):
  start, end = get_fold(len(eval_data), FLAGS.num_records, index)
  data = eval_data[start: end]
  ofile = f'{FLAGS.out_dir}/{FLAGS.mark}/record_{index}.tfrec'
  total = len(data)
  writer = melt.tfrecords.Writer(ofile)
  merger = Merger(writer)
  for instance in tqdm(bert_helper.convert(data), total=total):
    feature = {}
    if instance['qid'] == -1:
      continue
    for key in instance:
      if key not in set(['token_to_orig_map', 'doc_tokens', 'tokens', 'prev_is_whitespace_flag']):
        feature[key] = instance[key]
    merger(feature)
      
def main(_):
  np.random.seed(FLAGS.seed_)

  FLAGS.out_dir = f'{FLAGS.out_dir}/{FLAGS.record_name}'
  gezi.try_mkdir(f'{FLAGS.out_dir}/{FLAGS.mark}')

  data_folder = '/home/featurize/data/mrc/short/train_data/'
  dev_file = data_folder + FLAGS.ifile
  reader = NewDataReader(max_padding_len=10)

  global eval_data
  eval_data = reader.fast_read(dev_file, suffix=".gn.pkl")
  bert_dir = '/home/featurize/data/mrc/short/pretrained/chinese_roberta/'

  global bert_helper
  bert_helper = BertDataHelperForSQA(bert_dir, answer_position_as_list=False)

  with Pool(FLAGS.num_records) as p:
    p.map(build_features, range(FLAGS.num_records))

  if FLAGS.mark == 'dev':
    res = []
    for instance in tqdm(bert_helper.convert(eval_data), total=len(eval_data)):
      res.append(instance)
    gezi.save_pickle(res, f'../input/dev.pkl')

if __name__ == '__main__':
  flags.DEFINE_string('ifile', 'dev.lat.json', '')
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'dev', 'train or dev')
  flags.DEFINE_integer('num_records', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('record_name', 'tfrecords2', '')
  flags.DEFINE_integer('max_len', 512, '')
  
  app.run(main) 
