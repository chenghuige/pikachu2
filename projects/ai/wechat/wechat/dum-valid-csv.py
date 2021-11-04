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

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
from collections import OrderedDict
import pandas as pd
import melt as mt
import gezi
from gezi import tqdm

from wechat.dataset import Dataset
from wechat.config import *

def main(_):
  mark = 'valid' if FLAGS.mode != 'test' else 'test'
  input_pattern = f'../input/{FLAGS.records_name2}/{mark}/*.tfrec'
  ic(input_pattern)
  files=gezi.list_files(input_pattern)
  ic(len(files), files[:5])
  dataset = Dataset('valid', files=files)
  datas = dataset.make_batch(512)
  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  m = OrderedDict()
  for key in eval_keys:
    m[key] = []
  for x, y in tqdm(datas, total=num_steps):
    for key in eval_keys:
      m[key].extend(list(gezi.squeeze(x[key].numpy())))

  df = pd.DataFrame(m)
  ic(df.head())
  df.to_csv(f'../input/{mark}.csv', index=False)


if __name__ == '__main__':
  app.run(main)  
  
