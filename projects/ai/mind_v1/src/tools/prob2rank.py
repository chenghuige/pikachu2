#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   to-truth.py
#        \author   chenghuige  
#          \date   2020-08-22 02:23:32.058144
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
from tqdm import tqdm
import json

ifile = sys.argv[1]
ofile = ifile.replace('.csv', '_rank.csv')
print('prob2rank result:', ofile)
total =len(open(ifile).readlines())
with open(ofile, 'w') as out:
  for line in tqdm(open(sys.argv[1]), total=total, ascii=True):
    l = line.strip('\n').split(' ')
    impression_id, scores = l[0], l[-1]
    scores = json.loads(scores)
    ranks = (-np.asarray(scores)).argsort().argsort() + 1
    print(impression_id, '[' + ','.join(map(str, ranks)) + ']', sep=' ', file=out)
  
  
