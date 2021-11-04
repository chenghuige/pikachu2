#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-tfrecord.py
#        \author   chenghuige  
#          \date   2019-09-11 11:00:01.818073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
from gezi.util import index
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import pandas as pd

import melt as mt
import gezi
from gezi import tqdm

def main(_):
  tag_counter, cat_counter, subcat_counter = gezi.WordCounter(), gezi.WordCounter(), gezi.WordCounter()
  
  record_files = []
  dirs = ['../input/pointwise', '../input/pairwise', '../input/test_a/', '../input/test_b']
  tdirs = tqdm(dirs)
  for dir in tdirs:
    ids = []
    tdirs.set_postfix({'dir': dir})
    record_files = gezi.list_files(f'{dir}/*.tfrecords')
    # ic(record_files)
    t = tqdm(record_files)
    for record_file in t:
      t.set_postfix({'file': record_file})
      for i, item in enumerate(tf.data.TFRecordDataset(record_file)):
        x = mt.decode_example(item)
        if 'tag_id' in x:
          tag_ids = x['tag_id']
          tag_counter.adds(tag_ids)
        if 'category_id' in x:
          category_id = x['category_id'][0]
          if category_id != -1:
            cat = category_id // 100
            subcat = category_id 
            cat_counter.add(cat)
            subcat_counter.add(subcat)

  tag_counter.save('../input/tag_vocab.txt')
  cat_counter.save('../input/cat_vocab.txt')
  subcat_counter.save('../input/subcat_vocab.txt')

if __name__ == '__main__':
  app.run(main)  
  
