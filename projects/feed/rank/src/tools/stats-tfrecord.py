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
import six
import struct

import melt
import gezi
logging = gezi.logging
import tensorflow as tf

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('odir', None, '')
flags.DEFINE_integer('max_steps', None, '')

from projects.feed.rank.src.tfrecord_dataset import Dataset

# python stats-tfrecord.py  /search/odin/publicData/CloudS/yuwenmengke/rank_0804_so/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020062422
# mktest_user_kw_feed 455 93.18665318083667
# tw_history 102 51.514601080448315
# tw_history_rec 102 51.514601080448315
# tw_history_kw 408 206.05840432179326
# index 138 78.12678262107586
# mktest_doc_kw_secondary_feed 8 1.6048612264591193
# tw_history_topic 102 51.514601080448315
# vd_history 101 32.31048948421644
# value 138 78.12678262107586
# field 138 78.12678262107586
# doc_keyword 13 3.7979804870717953
# vd_history_topic 101 32.31048948421644
# mktest_new_search_kw_feed 279 2.539914695204423
# mktest_doc_kw_feed 11 3.1931192606126757

def filter_padding(l):
  if len(l) == 0:
    return l
  len_ = len(l)
  for i in reversed(range(len_)):
    if l[i] != 0:
      break
  return l[:(i + 1)]

def deal(files):
  batch_size, odir = FLAGS.batch_size, FLAGS.odir
  dataset = Dataset('valid', sparse_to_dense=True)
  batches = dataset.make_batch(batch_size=batch_size, filenames=files, repeat=False)
  total = melt.get_num_records(files)
  num_steps = -int(-total // (batch_size))
  if FLAGS.max_steps:
    num_steps = min(num_steps, FLAGS.max_steps)
  m, m2, counts = {}, {}, {}
  dynamics = set()
  for i, (x, _) in tqdm(enumerate(batches), total=num_steps, ascii=True, desc='clone'):
    if i == num_steps:
      break
    for key in x:
      x[key] = x[key].numpy()
    keys = list(x.keys())
    bs = x[keys[0]].shape[0]
    for i in range(bs):
      feature = {}
      for key in x:
        dtype = x[key].dtype
        if dtype not in [np.float32, np.float64, np.int32, np.int64]:
          continue
        try:
          l = filter_padding(x[key][i])
          if key not in m:
            m[key] = len(l)
            m2[key] = m[key]
            counts[key] = 1
          else:
            counts[key] += 1
            m2[key] += len(l)
            if len(l) > m[key]:
              m[key] = len(l)
              dynamics.add(key)
        except Exception:
          pass
  
  # for key in dynamics:
  for key in m:
    if (m[key] == m2[key] / counts[key]):
      print(key, m[key])
  for key in m:
    if (m[key] != m2[key] / counts[key]):
      print(key, m[key], m2[key] / counts[key])

def main(_):  
  in_dir = sys.argv[1]
  hour = os.path.basename(in_dir)
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  FLAGS.all_varlen_keys = True
  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  tf.compat.v1.enable_eager_execution()

  deal(files)


if __name__ == '__main__':
  app.run(main)
  
