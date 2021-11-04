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
FLAGS = flags.FLAGS

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
from collections import OrderedDict
import pandas as pd
import melt as mt
import gezi
from gezi import tqdm

from qqbrowser.dataset import Dataset
from qqbrowser import config
from qqbrowser.config import *

from baseline.tensorflow.config import parser
from baseline.tensorflow.data_helper import FeatureParser

def main(_):  
  # FLAGS.recount_tfrecords = True
  FLAGS.mode = FLAGS.mode or 'valid'
  marks = {
    'train': 'pointwise',
    'valid': 'pairwise',
    'test': 'test_a'
  }
  mark = marks[FLAGS.mode] if FLAGS.parse_strategy == 1 else f'{FLAGS.records_name}/{FLAGS.mode}'
  record_files = gezi.list_files(f'../input/{mark}/*.tfrec*')
  ic(record_files[:2])
  t = tqdm(record_files)
  for record_file in t:
    t.set_postfix({'file': record_file})
    for i, item in enumerate(tf.data.TFRecordDataset(record_file)):
      x = mt.decode_example(item)
      if i == 0:
        ic(list(x.keys()))
        # ic(len(x['frame_feature']))
        # ic(tf.io.decode_raw(x['frame_feature'], out_type=tf.float16))
      for key in x:
        ic(i, key, x[key].shape)
      if i == 2:
        break
    break
  
  config.init()

  if FLAGS.parse_strategy > 2:
    record_files = FLAGS.valid_files

  dataset = Dataset(FLAGS.mode, files=record_files)
  datas = dataset.make_batch(2)
  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  for x, y in tqdm(datas, total=num_steps):
    if isinstance(x, (list, tuple)):
      x = x[0]
    ic(list(x.keys()))
    for key in x:
      ic(key, x[key].dtype, x[key].shape)
    ic(x, y)
    break

  with gezi.Timer('loop dataset'):
    dataset = Dataset(FLAGS.mode, files=record_files)
    datas = dataset.make_batch(256)
    ic(dataset.num_instances)
    num_steps = dataset.num_steps
    for x, _ in tqdm(datas, total=num_steps):
      pass


if __name__ == '__main__':
  app.run(main)  
  
