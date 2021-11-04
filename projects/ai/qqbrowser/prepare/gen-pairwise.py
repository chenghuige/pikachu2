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

flags.DEFINE_integer('num_records', 40, '')
flags.DEFINE_bool('debug_', False, '')

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
from collections import OrderedDict
from multiprocessing import Pool, Manager, cpu_count

import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import scipy
import melt as mt
import gezi
from gezi import tqdm

from qqbrowser.dataset import Dataset
from qqbrowser import config
from qqbrowser.config import *

records = {}
df = None
mark = None
dfs = []

# TODO for each input label.csv using multiple workeers like 20, so faster especially for train
def build(index):
  d = df

  # label norm method from guodaya
  qt = QuantileTransformer(n_quantiles=100, random_state=0)
  d['relevance2'] = qt.fit_transform(d[['relevance']])

  # label norm method from https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st
  target = scipy.stats.rankdata(d.relevance, 'average')
  d['relevance3'] = (target - target.min()) / (target.max() - target.min())

  ofile = f'../input/{FLAGS.records_name}/pairwise/{mark}/{index}.tfrec'
  with mt.tfrecords.Writer(ofile, buffer_size=1000, shuffle=True, seed=1024) as writer:
    for i, row in tqdm(enumerate(d.itertuples()), total=len(d)):
      if i % FLAGS.num_records != index:
        continue
      fe = {}
      row = row._asdict()
      # if row['query'] not in records or row['candidate'] not in records:
      #   continue
      r1 = records[row['query']]
      r2 = records[row['candidate']]
      
      for key in r1:
        fe[f'{key}1'] = r1[key]
        fe[f'{key}2'] = r2[key]
      fe['relevance'] = row['relevance']
      fe['label'] = THRES[int(fe['relevance'] * 20)]
      fe['relevance2'] = row['relevance2']
      fe['relevance3'] = row['relevance3']

      fe['group'] = index
      fe['id'] = row['id']
      fe['pid'] = row['pid']

      fe['type_weight'] = 1.
      fe['relevance_weight'] = 1.
      if 'ori' in row:
        fe['ori'] = int(row['ori'])
      else:
        fe['ori'] = 1

      if not fe['ori']:
        fe['type_weight'] = 0.1

      if (not fe['ori']) and (fe['relevance'] == 1.):
        fe['relevance_weight'] = 0.3
      
      writer.write(fe)
    print(index, writer.num_records)

def main(_):  
  FLAGS.buffer_size = 100
  config.init()
  global df, dfs, mark
  dfs = [
    pd.read_csv('../input/pairwise/label_valid0.csv'),
    pd.read_csv('../input/pairwise/label_train0.csv'),
    pd.read_csv('../input/pairwise/label.csv'),
    pd.read_csv('../input/pairwise/label_valid1.csv'),
    pd.read_csv('../input/pairwise/label_train1.csv'),
    pd.read_csv('../input/pairwise/label_valid2.csv'),
    pd.read_csv('../input/pairwise/label_train2.csv'),
    pd.read_csv('../input/pairwise/label_valid3.csv'),
    pd.read_csv('../input/pairwise/label_train3.csv'),
    pd.read_csv('../input/pairwise/label_valid4.csv'),
    pd.read_csv('../input/pairwise/label_train4.csv'),
    # pd.read_csv('../input/pairwise/label_new_train0.csv'),
    # pd.read_csv('../input/pairwise/label_new.csv'),
  ]

  marks = [
    'valid0',
    'train0',
    'all',
    'valid1',
    'train1',
    'valid2',
    'train2',
    'valid3',
    'train3',
    'valid4',
    'train4',
    # 'new_train0',
    # 'new',
  ]

  ic(len(dfs))
  for i in tqdm(range(len(dfs))):
    if 'train' in marks[i] or 'all' in marks[i]:
      dfs[i] = dfs[i].sample(frac=1, random_state=1024)
    ic(i, marks[i], len(dfs[i]))
  
  indir = f'../input/{FLAGS.records_name}/valid'
  record_files = gezi.list_files(f'{indir}/*.tfrec*')

  if FLAGS.debug_:
    record_files = record_files[:1]

  ic(record_files)
  ic(mt.get_num_records_from_dir(indir))
  ic(mt.get_num_records_from_dir(indir, recount=True))
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
    # ic(x)
    #   break
    # break

  # ic(list(list(records.values())[0].keys()))

  for i in tqdm(range(len(dfs))):  
    mark = marks[i]
    df = dfs[i]
    out_dir = f'../input/{FLAGS.records_name}/pairwise/{mark}'
    if os.path.exists(out_dir):
      ic(out_dir, 'exists! ignore, you may need to rm -rf manually before running')
      continue
    gezi.try_mkdir(out_dir)
    with Pool(FLAGS.num_records) as p:
      p.map(build, range(FLAGS.num_records))
    ic(out_dir, mt.get_num_records_from_dir(out_dir))
    ic(out_dir, mt.get_num_records_from_dir(out_dir, recount=True))

if __name__ == '__main__':
  app.run(main)  
  
