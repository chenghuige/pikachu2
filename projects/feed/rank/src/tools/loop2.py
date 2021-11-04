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

import melt
import gezi
logging = gezi.logging
import tensorflow as tf
from tfrecord_lite import decode_example 

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('out_docid', False, '')
flags.DEFINE_bool('out_abtestid', False, '')
flags.DEFINE_bool('out_online', False, '')
flags.DEFINE_bool('out_all', True, '')
flags.DEFINE_bool('out_min', False, '')
flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('ofile', None, '')

from projects.feed.rank.src.tfrecord_dataset import Dataset

# just show usage, much slower then loop.py just use loop.py

def main(_):  
  in_dir = sys.argv[1]
  hour = os.path.basename(in_dir)
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files)
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  assert FLAGS.ofile
  if FLAGS.ofile and gezi.non_empty(os.path.realpath(FLAGS.ofile)):
    df = pd.read_csv(FLAGS.ofile)
    if len(df) == total:
      print('infos file %s exits do nothing' % FLAGS.ofile, file=sys.stderr)
      exit(1)
    else:
      print('num_done:', len(df), file=sys.stderr)
  else:
    print('write to', FLAGS.ofile, file=sys.stderr)

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  def get_item(files):
    for file in files:
      for it in tf.compat.v1.python_io.tf_record_iterator(file):
        yield it

  m = defaultdict(list)
  for it in tqdm(get_item(files), total=total):
    x = decode_example(it)
    keys = list(x.keys())
    for key in keys:
      if len(x[key]) != 1:
        del x[key]
        continue
      val = x[key][0]
      if type(val) == bytes:
        val = val.decode()
      m[key] += [val]
    
  df = pd.DataFrame(m)
  df['hour'] = hour
  assert len(df) == total
  df.to_csv(FLAGS.ofile, index=False)


if __name__ == '__main__':
  app.run(main)
  