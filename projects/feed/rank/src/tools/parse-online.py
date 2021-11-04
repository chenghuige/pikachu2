#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   parse-online.py
#        \author   chenghuige  
#          \date   2019-10-14 21:27:18.444032
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

testids = None

if len(sys.argv) > 1:
  if sys.argv[1] != '0':
    testids = set(sys.argv[1].split(','))

use_ori_lr_score = True
if len(sys.argv) > 2:
  use_ori_lr_score = False

count = 0
num_bads = 0
# mid, docid, abtestid, show_time, duration, pred, pred2
for i, line in enumerate(sys.stdin):
  l = line.rstrip().split(',')
  mid = l[0]
  abtestid = l[2]
  duration = l[4]
  if float(l[5]) < 0:
    num_bads += 1
    continue
  pred = l[5] if use_ori_lr_score else l[6]
  if testids and abtestid not in testids:
    continue
  count += 1
  print(mid, duration, pred, sep=',')

print('testid ratio:', count / i, file=sys.stderr)
print('bads ratio:', num_bads / count, file=sys.stderr)
