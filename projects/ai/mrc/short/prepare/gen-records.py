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

def build_features(index):
  start, end = gezi.get_fold(len(eval_data), FLAGS.num_records, index)
  data = eval_data[start: end]
  ofile = f'{FLAGS.out_dir}/{FLAGS.mark}/record_{index}.tfrec'
  total = len(data)
  with melt.tfrecords.Writer(ofile) as writer:
    for instance in tqdm(bert_helper.convert(data), total=total):
      feature = {}
      if instance['qid'] == -1:
        continue
      for key in instance:
        if key not in set(['token_to_orig_map', 'doc_tokens', 'tokens', 'prev_is_whitespace_flag']):
          feature[key] = instance[key]
      writer.write_feature(feature)

def main(_):
  np.random.seed(FLAGS.seed_)

  FLAGS.out_dir = f'{FLAGS.out_dir}/{FLAGS.record_name}'
  gezi.try_mkdir(f'{FLAGS.out_dir}/{FLAGS.mark}')

  data_folder = '/home/featurize/data/mrc/short/train_data/'
  dev_file = data_folder + FLAGS.ifile
  reader = NewDataReader(max_padding_len=10)

  global eval_data
  eval_data = reader.fast_read(dev_file, suffix=".gn.pkl")
  # eval_data = []
  # for line in open(dev_file):
  #   x = json.loads(line.strip())
  #   if 'is_short' in x and x['is_short'] == 0:
  #       continue
    
  #   for passage in x['passages']:
  #     for key in x:
  #       if key == 'passages':
  #         continue
  #       passage[key] = x[key]
  #     eval_data.append(passage)

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
  # flags.DEFINE_string('ifile', 'dev.lat.json', '')
  flags.DEFINE_string('ifile', 'test_data.json', '')
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'dev', 'train or dev')
  flags.DEFINE_integer('num_records', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('record_name', 'tfrecords', '')
  flags.DEFINE_integer('max_len', 512, '')
  
  app.run(main) 
