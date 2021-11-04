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

ofile = sys.argv[2]
print('ofile', ofile, file=sys.stderr)

in_dir = sys.argv[1]
files = gezi.list_files(in_dir)
total = melt.get_num_records(files)

def get_item(files):
  for file in files:
    for it in tf.compat.v1.python_io.tf_record_iterator(file):
      yield it

shows = defaultdict(int)
clicks = defaultdict(int)
durations = defaultdict(int)

for it in tqdm(get_item(files), total=total):
  x = decode_example(it)
  click = x['click'][0]
  duration = x['duration'][0]
  if duration > 60 * 60 * 12:
    duration = 60
  id = x['id'][0].decode()
  uid, _ = id.split('\t')

  shows[uid] += 1
  clicks[uid] += click  
  durations[uid]+= duration  

num_users = len(shows)
num_users_noclick = len([x for x in clicks.values() if x == 0])
num_shows = sum(shows.values())
num_clicks = sum(clicks.values())
total_duration = sum(durations.values())
print('total users', num_users, 
      'num users no click', num_users_noclick,
      'total clicks', num_clicks, 
      'click rate', num_clicks / num_shows,
      'clicks per user', num_clicks / num_users, 
      'duration per user', total_duration / num_users,
      'duration per click', total_duration / num_clicks, 
      'duration per show', total_duration / num_shows,
      file=sys.stderr
     )

with open(ofile, 'w') as out:
  for user in shows.keys():
    print(user, shows[user], clicks[user], durations[user], sep='\t', file=out)
    
