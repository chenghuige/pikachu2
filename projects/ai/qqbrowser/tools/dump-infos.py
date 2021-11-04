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

import sys 
import os

import numpy as np
import pandas as pd

import melt as mt
import gezi
from gezi import tqdm

def main(_):
  record_files = []
  dirs = ['../input/pointwise', '../input/pairwise', '../input/test_a', '../input/test_b']
  tdirs = tqdm(dirs)
  all = []
  for dir in tdirs:
    ms = []
    tdirs.set_postfix({'dir': dir})
    record_files = gezi.list_files(f'{dir}/*.tfrecords')
    # ic(record_files)
    t = tqdm(record_files)
    for record_file in t:
      t.set_postfix({'file': record_file})
      for i, item in enumerate(tf.data.TFRecordDataset(record_file)):
        x = mt.decode_example(item)
        id = gezi.decode(x['id'])[0]
        title = gezi.decode(x['title'])[0]
        asr_text = gezi.decode(x['asr_text'])[0]
        if 'tag_id' in x:
          tag_id = ','.join([str(x) for x in x['tag_id']])
        else:
          tag_id = np.nan

        if 'category_id' in x:
          category_id = x['category_id'][0]
          cat = int(category_id / 100)
          subcat = int(category_id % 100)
        else:
          category_id = np.nan
          cat = np.nan
          subcat = np.nan

        m = {
          'id': id,
          'category_id': category_id,
          'cat': cat,
          'subcat': subcat,
          'tag_id': tag_id,
          'title': title,
          'asr_text': asr_text
        }
        ms.append(m)
        all.append(m)

        # ic(ms)

    if ms:
      df = pd.DataFrame(ms)
      df.to_csv(f'{dir}/infos.csv', index=False)
    
  if all:
    df = pd.DataFrame(all)
    df.to_csv(f'../input/infos.csv', index=False)

if __name__ == '__main__':
  app.run(main)  
  
