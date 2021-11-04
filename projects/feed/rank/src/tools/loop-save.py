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
import pickle

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('ofile', None, '')
flags.DEFINE_bool('title', False, '')

from projects.feed.rank.src.tfrecord_dataset import Dataset

#  python loop-save.py /search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520,/search/odin/publicData/CloudS/libowei/rank4/newmse/data/video_hour_newmse_v1/tfrecords/2020051520,/search/odin/publicData/CloudS/libowei/rank4/shida/data/video_hour_shida_v1/tfrecords/2020051520

def main(_):  
  in_dir = sys.argv[1]
  files = gezi.list_files(in_dir)
  total = melt.get_num_records(files) 
  print('total', total, file=sys.stderr)

  if not total:
    exit(1)

  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

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
      x[key] = x[key].numpy()
      if not len(x[key]):
        del x[key]
        continue
      m[key] += list(x[key])

  for key in tqdm(m.keys()):
    try:
      m[key] = np.concatenate(m[key], 0)
    except Exception:
      # print(m[key])
      print(key)

  with gezi.Timer('save--', print_fn=print, print_before=True) as timer:
    # np.save('/home/gezi/data/rank/video.npy', m)
    pickle.dump(m,open('/home/gezi/data/rank/video.npy','wb'),protocol = 4)


if __name__ == '__main__':
  app.run(main)
  
