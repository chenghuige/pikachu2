#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   get-all-uers.py
#        \author   chenghuige  
#          \date   2019-08-18 11:06:39.496266
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import collections
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import traceback
import pandas as pd
from itertools import repeat
from multiprocessing import Pool, Manager, cpu_count
import pymp

import melt
import gezi
logging = gezi.logging
import tensorflow as tf

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('odir', None, '')

from projects.feed.rank.src.tfrecord_dataset import Dataset

def filter_padding(l):
  if len(l) == 0:
    return l
  len_ = len(l)
  for i in reversed(range(len_)):
    if l[i] != 0:
      break
  return l[:(i + 1)]

def gen_feature(l, dtype):
  l = list(filter_padding(l))
  if dtype == np.int64:
    return melt.int64_feature(l)
  elif dtype == np.float32 or dtype == np.float64:
    return melt.float_feature(l)
  elif dtype == np.object or dtype == np.str_:
    return melt.bytes_feature(l)
  else:
    raise TypeError(dtype)

fid2field = {}

def modify_example(x):
  l = []
  indexes = x['index']
  values = x['value']
  fields = x['field']
  bs = len(x['index'])
  for field in fid2field.values():
    x[f'index_{field}'] = [np.asarray([0], dtype=np.int64)] * bs
    x[f'value_{field}'] = [np.asarray([0.], dtype=np.float32)] * bs
  for i in range(bs):
    field_index = defaultdict(list)
    field_value = defaultdict(list)
    for fid, index, value in zip(fields[i], indexes[i], values[i]):
      if fid == 0:
        break
      field_index[fid2field[fid]].append(index)
      if value == 0. and index != 0:
        value = 1.
      field_value[fid2field[fid]].append(value)

    for field in field_index:
      x[f'index_{field}'][i] = np.asarray(field_index[field], dtype=np.int64)

    for field in field_value:
      x[f'value_{field}'][i] = np.asarray(field_value[field], dtype=np.float32)
  
  del x['index']
  del x['value']
  del x['field']

  return x

def deal(file, batch_size, odir):
  dataset = Dataset('valid', sparse_to_dense=True)
  batches = dataset.make_batch(batch_size=batch_size, filenames=[file], repeat=False)
  total = melt.get_num_records([file])
  num_steps = -int(-total // (batch_size))
  ofile = os.path.join(odir, os.path.basename(file))
  with melt.tfrecords.Writer(ofile, 1000) as writer:
    for i, (x, _) in tqdm(enumerate(batches), total=num_steps, ascii=True, desc='clone'):
      for key in x:
        x[key] = x[key].numpy()
      x = modify_example(x)
      keys = list(x.keys())
      bs = x[keys[0]].shape[0]
      for i in range(bs):
        feature = {}
        for key in x:
          if isinstance(x[key], list):
            dtype = x[key][i].dtype
          else:
            dtype = x[key].dtype
          if isinstance(x[key], list):
            feature[key] = gen_feature(x[key][i], dtype)   
          elif len(x[key].shape) == 1:
            feature[key] = gen_feature([x[key][i]], dtype)
          elif x[key].shape[1] == 0:
            feature[key] = gen_feature([], dtype)
          else:
            feature[key] = gen_feature(x[key][i], dtype)                
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(record)

def main(_):  
  in_dir = sys.argv[1]
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  config = '/home/gezi/mine/pikachu/projects/feed/rank/src/conf/tuwen/fields.hash' 
  if 'video' in in_dir:
    config = config.replace('tuwen', 'video')

  for line in open(config):
    field, fid = line.strip().split()
    fid2field[int(fid)] = field

  if not FLAGS.odir:
    FLAGS.odir = os.path.abspath(in_dir).replace('v1', 'v2')
  os.system(f'mkdir -p {FLAGS.odir}')

  print('write to', FLAGS.odir, file=sys.stderr)

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  tf.compat.v1.enable_eager_execution()
  
  # ps = min(cpu_count(), len(files))
  ps = len(files)
  with Pool(ps) as p:
    p.starmap(deal, [(file, batch_size, FLAGS.odir) for file in files])

if __name__ == '__main__':
  app.run(main)
  
