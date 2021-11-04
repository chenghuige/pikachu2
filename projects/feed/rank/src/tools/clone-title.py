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
import glob
import pickle
import pymp
import jieba

import melt
import gezi
logging = gezi.logging
import tensorflow as tf

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('odir', None, '')
flags.DEFINE_bool('title', False, '')

from projects.feed.rank.src.tfrecord_dataset import Dataset
from projects.feed.rank.utils.doc_info.KV import KV

titles = {}
docids = set()

def filter_padding(l):
  if len(l) == 0:
    return l
  len_ = len(l)
  for i in reversed(range(len_)):
    if l[i] != 0:
      break
  return l[:(i + 1)]

def gen_feature(l, dtype):
  # l = list(filter_padding(l))
  l = list(l)
  if dtype == np.int64:
    return melt.int64_feature(l)
  elif dtype == np.float32 or dtype == np.float64:
    return melt.float_feature(l)
  elif dtype == np.object or dtype == np.str_:
    return melt.bytes_feature(l)
  else:
    raise TypeError(dtype)

def modify_example(x):
  bs = len(x['index'])
  x['title'] = [np.asarray([0], dtype=np.int64)] * bs
  docids = gezi.decode(x['docid'])
  for i in range(bs):
    if docids[i] in titles:
      x['title'][i] = titles[docids[i]]
    for j in range(len(x['value'][i])):
      if x['value'][i][j] == 0.:
        x['value'][i][j] = 1.
  return x

def deal(file):
  batch_size, odir = FLAGS.batch_size, FLAGS.odir
  dataset = Dataset('valid', sparse_to_dense=True)
  batches = dataset.make_batch(batch_size=batch_size, filenames=[file], repeat=False)
  total = melt.get_num_records([file])
  num_steps = -int(-total // (batch_size))
  ofile = os.path.join(odir, os.path.basename(file))
  print(ofile)
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

def get_titles(files, total):
  batch_size = FLAGS.batch_size
  dataset = Dataset('valid')
  batches = dataset.make_batch(batch_size=FLAGS.batch_size, filenames=files, repeat=False)
  num_steps = -int(-total // batch_size)
  print('----num_steps', num_steps, file=sys.stderr) 

  global docids
  for i, (x, _) in tqdm(enumerate(batches), total=num_steps, ascii=True, desc='loop'):
    dids = set(gezi.decode(x['docid'].numpy()))
    docids.update(dids)

  kv = KV()
  titles_ = kv.get_titles(list(set(docids)))

  for did, title in titles_.items():
    titles[did] = np.asarray([gezi.hash_int64(x) for x in jieba.cut(title)])

def main(_):  
  in_dir = sys.argv[1]
  hour = os.path.basename(in_dir)
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  if not FLAGS.odir:
    FLAGS.odir = os.path.abspath(in_dir).replace('yuwenmengke', 'chenghuige')

  assert FLAGS.odir
  os.system(f'mkdir -p {FLAGS.odir}')

  print('write to', FLAGS.odir, file=sys.stderr)

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  tf.compat.v1.enable_eager_execution()

  global titles
  ## TODO strange will hange for multipleprocess if use dataset here
  get_titles(files, total)
  # if FLAGS.title:
  #   get_titles(files, total)
  #   # print(titles)
  #   os.system('mkdir -p /home/gezi/tmp/titles')
  #   with open(f'/home/gezi/tmp/titles/{hour}.pkl', 'wb') as f:
  #     pickle.dump(titles, f)
  #   exit(0)
  # else:
  #   with open(f'/home/gezi/tmp/titles/{hour}.pkl', 'rb') as f:
  #     titles = pickle.load(f)
  
  ps = len(files)

  print('---ps', ps)
  # with Pool(ps) as p:
  #   p.map(deal, files)
  with pymp.Parallel(ps) as p:
    for i in tqdm(p.range(ps), desc='deal', ascii=True):
      deal(files[i])


if __name__ == '__main__':
  app.run(main)
  
