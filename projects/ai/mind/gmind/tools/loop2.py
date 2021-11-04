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

flags.DEFINE_integer('batch_size_', 1024, '')
flags.DEFINE_bool('title', False, '')

from projects.ai.mind.src.dataset import Dataset

def main(_):  
  in_dir = sys.argv[1]
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  ofile = sys.argv[2]
  df = None
  if ofile and gezi.non_empty(os.path.realpath(ofile)):
    try:
      df = pd.read_csv(ofile)
      if len(df) == total and (not FLAGS.title or 'title' in df.columns):
        print(f'infos file {ofile} exits do nothing', file=sys.stderr)
        exit(0)
      else:
        print('num_done:', len(df), file=sys.stderr)
    except Exception:
      pass
  
  print('write to', ofile, file=sys.stderr)

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  if tf.__version__ < '2':
    tf.compat.v1.enable_eager_execution()
  
  dataset = Dataset('valid')
  print('---batch_size', dataset.batch_size, FLAGS.batch_size, melt.batch_size(), file=sys.stderr)  
  
  batches = dataset.make_batch(batch_size=batch_size, filenames=files, repeat=False)

  num_steps = -int(-total // batch_size)
  print('----num_steps', num_steps, file=sys.stderr) 
  m = defaultdict(list)
  for i, (x, _) in tqdm(enumerate(batches), total=num_steps, ascii=True, desc='loop'):
    bs = len(x['id'])
    keys = list(x.keys())
    for key in keys:
      if key not in ['impression_id', 'did_']:
        continue
      x[key] = x[key].numpy()
      if not len(x[key]):
        del x[key]
        continue
      if x[key].shape == (bs, 1):
        x[key] = gezi.squeeze(x[key])
      if x[key].shape != (bs,):
        del x[key]
        continue
      if x[key].dtype == np.object:
        x[key] = gezi.decode(x[key])
      m[key] += [x[key]]
    if i == 0:
      if df is not None and len(df) == total and set(m.keys()) == set(list(df.columns)):
        print(f'infos file {ofile} exits do nothing', file=sys.stderr)
        exit(0)

  for key in m.keys():
    m[key] = np.concatenate(m[key], 0)
    
  df = pd.DataFrame(m)
  df.to_csv(ofile, index=False)


if __name__ == '__main__':
  app.run(main)
  
