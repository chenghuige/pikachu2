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
import numpy as np

ofile = sys.argv[2]
print('ofile', ofile, file=sys.stderr)

in_dir = sys.argv[1]
files = gezi.list_files(in_dir)
total = melt.get_num_records(files)

def get_item(files):
  for file in files:
    for it in tf.compat.v1.python_io.tf_record_iterator(file):
      yield it

uinfo = {}
uids = []
for it in tqdm(get_item(files), total=total):
  x = decode_example(it)
  duration = x['duration'][0]
  if duration > 60 * 60 * 12:
    duration = 60
  id = x['id'][0].decode()
  uid, doc_id = id.split('\t')
  
  info = '{}:{}'.format(doc_id, duration)
  if uid in uinfo:
    uinfo[uid].append(info)
  else:
    uinfo[uid] = [info]
    uids.append(uid)

np.random.shuffle(uids)
    
with open(ofile, 'w') as out:
  for uid in uids:
    info = uinfo[uid]
    print('{}\t{}'.format(uid, ','.join(info)), file=out)
