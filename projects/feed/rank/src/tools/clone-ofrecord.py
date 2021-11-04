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

import oneflow.core.record.record_pb2 as ofrecord

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

from projects.feed.rank.src.tfrecord_dataset import Dataset

lens = {}

fields = {}

# python clone-ofrecord.py /search/odin/publicData/CloudS/yuwenmengke/rank_0804_so/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020062422

def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))

def float_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))

def double_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))

def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))

def padding(l, key):
  if len(l) == 0:
    return l
  if key and key in lens:
    return list(gezi.pad(l, lens[key]))
  else:
    return list(l)

def gen_feature(l, dtype, key=None):
  l = padding(l, key)
  if key == 'field': 
    l = [fields[x] if x in fields else 0 for x in l]
  try:
    if dtype == np.int64:
      return int64_feature(l)
    elif dtype == np.float32 or dtype == np.float64:
      return float_feature(l)
    elif dtype == np.object or dtype == np.str_:
      return bytes_feature(l)
    else:
      raise TypeError(dtype)
  except Exception:
    print(key, l)

files = None
def deal(i):
  file = files[i]
  batch_size, odir = FLAGS.batch_size, FLAGS.odir
  dataset = Dataset('valid', sparse_to_dense=True)
  batches = dataset.make_batch(batch_size=batch_size, filenames=[file], repeat=False)
  total = melt.get_num_records([file])
  num_steps = -int(-total // (batch_size))
  ofile = os.path.join(odir, f'part-{i}')
  with open(ofile, 'wb') as f:
    for i, (x, _) in tqdm(enumerate(batches), total=num_steps, ascii=True, desc='clone'):
      for key in x:
        x[key] = x[key].numpy()
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
            feature[key] = gen_feature(x[key][i], dtype, key)   
          elif len(x[key].shape) == 1:
            feature[key] = gen_feature([x[key][i]], dtype, key)
          elif x[key].shape[1] == 0:
            feature[key] = gen_feature([], dtype, key)
          else:
            feature[key] = gen_feature(x[key][i], dtype, key)   
          
        ofrecord_features = ofrecord.OFRecord(feature=feature)
        serilizedBytes = ofrecord_features.SerializeToString()
        length = ofrecord_features.ByteSize()
        f.write(struct.pack("q", length))
        f.write(serilizedBytes)

def main(_):  
  in_dir = sys.argv[1]
  hour = os.path.basename(in_dir)
  global files
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  if not FLAGS.odir:
    FLAGS.odir = os.path.abspath(in_dir).replace('tfrecords', 'ofrecords')

  assert FLAGS.odir
  if 'ofrecords' in FLAGS.odir:
    os.system(f'rm -rf {FLAGS.odir}')
    os.system(f'mkdir -p {FLAGS.odir}')

  print('write to', FLAGS.odir, file=sys.stderr)

  mark = 'video' if 'video' in in_dir else 'tuwen'
  for line in open(f'../conf/{mark}/varlen.txt'):
    l = line.strip().split()
    key, val = l[0], l[1]
    val = int(l[1])
    lens[key] = val

  i = 1
  for line in open(f'../conf/{mark}/fields2.txt'):
    fid = int(line.strip())
    fields[fid] = i
    i += 1
  fields[0] = 0

  print('num_fields:', len(fields))

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  tf.compat.v1.enable_eager_execution()

  ps = len(files)
  with pymp.Parallel(ps) as p:
    for i in tqdm(p.range(ps), desc='deal', ascii=True):
      deal(i)

  print(f'{FLAGS.odir} done')


if __name__ == '__main__':
  app.run(main)
  
