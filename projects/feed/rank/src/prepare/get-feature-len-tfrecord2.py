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
import melt
import gezi
from tfrecord_lite import decode_example 
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
#tf.enable_v2_behavior()

in_dir = sys.argv[1]
files = gezi.list_files(in_dir)
total = melt.get_num_records(files)

## TODO slower compare to use iterator..
#@tf.function
def get_item(files):
  dataset = tf.data.TFRecordDataset(files)
  for raw_record in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    yield example

if len(sys.argv) > 2:
  ofile = sys.argv[2]
  out = open(ofile, 'w') 
else:
  out = sys.stdout


for example in tqdm(get_item(files), total=total):
  print(len(example.features.feature['index'].int64_list.value), file=out)
