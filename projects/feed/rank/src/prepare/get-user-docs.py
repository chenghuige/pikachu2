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

ofile = sys.argv[2]
print('ofile', ofile, file=sys.stderr)

uinfo2 = {}
uinfo = {}

for line in open(sys.argv[1]):
  x, duration = line.strip().split()
  duration = int(duration)
  uid, ts, doc_id = x.split('_', 2)
  if duration > 60 * 60 * 12:
    duration = 60

  id = '{}\t{}'.format(uid, doc_id)
  if id in uinfo2:
    if duration > uinfo2[id]:
      uinfo2[id] = duration
  else:
    uinfo2[id] = duration
  
uids = []
for id, duration in uinfo2.items():
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
