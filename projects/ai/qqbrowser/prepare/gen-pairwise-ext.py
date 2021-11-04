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

flags.DEFINE_integer('index', None, '')
flags.DEFINE_integer('num_records', 40, '')

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
from collections import OrderedDict
from multiprocessing import Pool, Manager, cpu_count

import pandas as pd
import melt as mt
import gezi
from gezi import tqdm

from qqbrowser.dataset import Dataset
from qqbrowser import config
from qqbrowser.config import *

records = {}
df = None
dfs = []
mark = None

# TODO for each input label.csv using multiple workeers like 20, so faster especially for train
def build(index):
  d = df
  ofile = f'../input/{FLAGS.records_name}/pairwise/{mark}/{index}.tfrec'
  with mt.tfrecords.Writer(ofile, buffer_size=1000, shuffle=True, seed=1024) as writer:
    for i, row in tqdm(enumerate(d.itertuples()), total=len(d)):
      if i % FLAGS.num_records != index:
        continue
      row = row._asdict()
      if row['query'] not in records or row['candidate'] not in records:
        continue
      r1 = records[row['query']]
      r2 = records[row['candidate']]
      fe = {}
      for key in r1:
        fe[f'{key}1'] = r1[key]
        fe[f'{key}2'] = r2[key]
      fe['relevance'] = [row['relevance']]
      fe['group'] = [index]
      fe['id'] = [row['id']]
      fe['pid'] = [row['pid']]

      if 'ori' in row:
        fe['ori'] = int(row['ori'])
      else:
        fe['ori'] = 1
      writer.write(fe)

def main(_):  
  FLAGS.buffer_size = 100
  config.init()
  global df, dfs, mark
  dfs = [
    # pd.read_csv('../input/pairwise/label_new2_train0.csv'),
    pd.read_csv('../input/pairwise/label_new_train0.csv'),
    # pd.read_csv('../input/pairwise/label_new2.csv'),
    #pd.read_csv('../input/pairwise/label_new.csv'),
    # pd.read_csv('../input/pairwise/label_new2_train1.csv'),
    # pd.read_csv('../input/pairwise/label_new_train1.csv'),
  ]
  marks = [
    # 'new2_train0',
    'new_train0',
    # 'new2',
    #'new'
  ]
  ic(len(dfs))
  for i in tqdm(range(len(dfs))):
    dfs[i] = dfs[i].sample(frac=1, random_state=1024)
  
  record_files = gezi.list_files(f'../input/{FLAGS.records_name}/valid/*.tfrec*')
  ic(record_files)
  dataset = mt.Dataset('valid', files=record_files)
  datas = dataset.make_batch(512, return_numpy=True)  
  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  
  for x in tqdm(datas, total=num_steps):
    count = len(x['vid'])
    vids = gezi.decode(x['vid'])
    for i in range(count):
      m = {}
      for key in x:
        m[key] = list(x[key][i])
      records[int(vids[i])] = m 
    ic(x)
    #   break
    # break

  # ic(list(list(records.values())[0].keys()))
  
  gezi.try_mkdir(f'../input/{FLAGS.records_name}/pairwise/new2_train')
  gezi.try_mkdir(f'../input/{FLAGS.records_name}/pairwise/new_train')
 
  for i, df_ in tqdm(enumerate(dfs), total=len(dfs)):
    df = df_
    mark = marks[i]
    with Pool(FLAGS.num_records) as p:
      p.map(build, range(FLAGS.num_records))

if __name__ == '__main__':
  app.run(main)  
  
