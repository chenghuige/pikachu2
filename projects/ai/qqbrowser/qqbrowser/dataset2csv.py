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

def deal(l):
  if len(l) == 1:
    return l[0]
  # if len(l) > 1000:
  #   # return ','.join([str(x) for x in l[:5] + l[-5:]])
  #   return ','.join([str(l[i]) for i in range(2)])
  return ','.join([str(x) for x in l if x > 0])

def main(_):  
  FLAGS.mode = 'train'
  mark = f'{FLAGS.records_name}/{FLAGS.mode}'
  record_files = gezi.list_files(f'../input/{mark}/*.tfrec*')
  ic(record_files)
  excl_keys = ['frames']
  # excl_keys = []
  dataset = mt.Dataset('valid', files=record_files, excl_keys=excl_keys)
  datas = dataset.make_batch(4096, return_numpy=True)
  # datas = dataset.make_batch(4096, return_numpy=True, repeat=False, drop_remainder=False, shuffle=True)
  
  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  # keys = ['vid', 'title_ids', 'tag_id', 'frames']
  keys = ['vid', 'title_ids', 'tag_id', 'num_frames']
  
  l = []
  for x in tqdm(datas, total=num_steps):
    count = len(x[keys[0]])
    # ic(x['vid'])
    # x['vid'] = gezi.decode(x['vid']).reshape([-1, 1])
    # already melt.try_append_dim to [-1,1] in mt.Dataset.parse
    # x['vid'] = gezi.decode(x['vid'])
    # continue
    # ic(x['vid'][0])
    for i in range(count):
      m = {}
      for key in keys:
        m[key] = deal(list(x[key][i]))
        # m[key] = ','.join(map(str, x[key][i]))
      m['title_maxlen'] = len(x['title_ids'][i])
      m['tag_maxlen'] = len(x['tag_id'][i])
      m['title_len'] = len(m['title_ids'])
      m['tag_len'] = len(m['tag_id'])
      l.append(m)
    #   ic(m)
    #   break
    # break
    # if len(l) > 1000:
    #   break
  df = pd.DataFrame(l)
  df = df.sort_values(['vid'])
  ofile = f'../input/{FLAGS.records_name}_{FLAGS.mode}.csv'
  ic(ofile)
  df.to_csv(ofile, index=False)


if __name__ == '__main__':
  app.run(main)  
  
