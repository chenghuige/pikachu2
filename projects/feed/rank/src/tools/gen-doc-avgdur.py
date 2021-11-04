#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-doc-avgdur.pyIAVGDUR
#        \author   chenghuige  
#          \date   2019-08-24 12:56:18.296824
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
#from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

from text_dataset import Dataset 

qt = QuantileTransformer(n_quantiles=255, random_state=0)

scaler = MinMaxScaler()

dataset = Dataset()

count = 0
total = 0
vals = []

durs = []

for line in tqdm(sys.stdin, total=568098):
  l = line.rstrip().split('\t')
  id, click, dur = '{}\t{}'.format(l[2], l[3]), int(l[0]), int(l[1])
  l = l[4:]
  durs.append(dur)
  find = 0
  for item in l:
    idx, val = item.split(':')
    idx = int(idx)
    if dataset.feat_to_field_name[idx] == 'IAVGDUR':
      #print(id, click, dur, dataset.feat_to_field_val[idx])
      vals.append(int(dataset.feat_to_field_val[idx]))
      find += 1
      pass

  if not find:
    #print(id, click, dur)
    vals.append(0)
    count += 1
    find = 1
  
  total += 1
  assert find == 1, find

print(total, count)

vals = np.asarray(vals)
print(vals)

vals = vals.reshape(-1, 1)
transformer = RobustScaler().fit(vals)

print(qt.fit_transform(vals).reshape(-1))

vals2 = transformer.transform(vals).reshape(-1)

print(vals2)

qt.fit(vals)

vals3 = qt.transform(vals).reshape(-1)
print(vals3)

durs = np.asarray(durs)
print(vals.shape, durs.shape)
x = np.concatenate([vals.reshape(-1, 1), durs.reshape(-1, 1)], 1)
qt.fit(x)

print(x)
print(qt.transform(x))


# print(vals2)
# print((vals2 + 1) / 2)
# scaler.fit(vals2.reshape(-1, 1))

# print(scaler.transform(vals2.reshape(-1, 1)).reshape(-1))

qt.fit(durs.reshape(-1, 1))

print(qt.transform([[0]]))
print(qt.transform([[1]]))
print(qt.transform([[20]]))
print(qt.transform([[21]]))
print(qt.transform([[40]]))
print(qt.transform([[120]]))
print(qt.transform([[600]]))