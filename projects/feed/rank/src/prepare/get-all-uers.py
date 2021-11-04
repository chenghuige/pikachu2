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

ofile = sys.argv[1]
print('ofile', ofile, file=sys.stderr)

total = 37000000

shows = defaultdict(int)
clicks = defaultdict(int)
durations = defaultdict(int)

for i, line in tqdm(enumerate(sys.stdin), total=total, ascii=True):
  l = line.rstrip().split('\t')
  click = int(l[0])
  duration = int(l[1])
  if duration > 60 * 60 * 12:
    duration = 60
  uid = l[2]

  shows[uid] += 1
  clicks[uid] += click  
  durations[uid]+= duration

  # if i == 10000:
  #   break

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

