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

# from qqbrowser.dataset import Dataset
# from qqbrowser.config import *
from baseline.tensorflow.config import parser
from baseline.tensorflow.data_helper import FeatureParser

def main(_):
  FLAGS.batch_size = 256
  mark = 'valid' if FLAGS.mode != 'test' else 'test'
  input_pattern = f'../input/pairwise/*.tfrecords'
  ic(input_pattern)
  files=gezi.list_files(input_pattern)
  ic(len(files), files[:5])

  args = parser.parse_args()
  fparser = FeatureParser(args)
  dataset = fparser.create_dataset(files, training=False, batch_size=FLAGS.batch_size)
 
  with gezi.Timer('loop dataset'):
    for x in tqdm(dataset, total=10000):
      # ic(x)
      # break
      pass


if __name__ == '__main__':
  app.run(main)  
  
