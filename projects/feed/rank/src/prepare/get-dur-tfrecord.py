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

in_dir = sys.argv[1]
files = gezi.list_files(in_dir)
total = melt.get_num_records(files)

def get_item(files):
  for file in files:
    for it in tf.compat.v1.python_io.tf_record_iterator(file):
      yield it

ofile = sys.argv[2]

with open(ofile, 'w') as out:
  for it in tqdm(get_item(files), total=total):
    x = decode_example(it)
    print((x['duration'][0]), file=out)
