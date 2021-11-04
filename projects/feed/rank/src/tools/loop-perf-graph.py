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

def main(_):  
  in_dir = sys.argv[1]
  hour = os.path.basename(in_dir)
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size 

  # with tf.device('/cpu:0'):
  melt.init(tf.Graph())
  sess = melt.get_session()
  
  with sess.graph.as_default():
    dataset = Dataset('valid')
    iter = dataset.make_batch(batch_size=batch_size, filenames=files, repeat=False)
    op = iter.get_next()
    print('---batch_size', dataset.batch_size, FLAGS.batch_size, melt.batch_size(), file=sys.stderr)  
    num_steps = -int(-total // batch_size)
    print('----num_steps', num_steps, file=sys.stderr) 
    l = []
    for i in tqdm(range(num_steps), ascii=True, desc='loop'):
      x, _ = sess.run(op)
      l.append(x['id'])
    print(len(l), file=sys.stderr)

if __name__ == '__main__':
  app.run(main)
  
