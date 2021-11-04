#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   online-infer.py
#        \author   chenghuige  
#          \date   2019-09-26 00:11:11.370410
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import subprocess
from tqdm import tqdm

total = sum(1 for _ in open(sys.argv[1]))

max_count = 1000
if len(sys.argv) > 3:
  max_count = int(sys.argv[3])
total = min(total, max_count)

ofile = '{}.{}.result'.format(sys.argv[1], sys.argv[2])

mids = set()
if os.path.exists(ofile):
  for line in open(ofile):
    mid = line.split()[0]
    mids.add(mid)

with open(ofile, 'a') as out:
  count = 0
  for line in tqdm(open(sys.argv[1]), total=total):
    mid, infos = line.rstrip().split('\t')
    if mid in mids:
      continue
    infos = infos.split(',')
    infos = [x.split(':') for x in infos]

    m = dict(infos)
    doc_ids, durs = list(zip(*infos))

    result = subprocess.check_output(["python", "src/RecThrift/PythonRecServerDemon/get_rank_test.py", sys.argv[2], mid, ','.join(doc_ids)])
    # print(result.strip().split('\n'))
    l = result.strip().split('\n')
    for item in l:
      doc_id, score = item.split()
      dur = int(m[doc_id])
      print(mid, doc_id, dur, int(dur > 0), score, file=out)
    count += 1
    if count == total:
      break


