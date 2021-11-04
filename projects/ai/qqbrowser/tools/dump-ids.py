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
  record_files = []
  dirs = ['../input/pointwise', '../input/pairwise', '../input/test_a/', '../input/test_b']
  tdirs = tqdm(dirs)
  all_ids = set()
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
        id = gezi.decode(x['id'])[0]
        ids.append(id)
        all_ids.add(id)
    if ids:
      df = pd.DataFrame({'id': ids})
      df.to_csv(f'{dir}/ids.csv', index=False)
    
  if all_ids:
    df = pd.DataFrame({'id': list(all_ids)})
    df.to_csv(f'../input/ids.csv', index=False)

if __name__ == '__main__':
  app.run(main)  
  
