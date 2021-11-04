#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics2tb.py
#        \author   chenghuige  
#          \date   2020-04-20 00:58:00.005064
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import gezi
import pandas as pd

idir = sys.argv[1]
os.system(f'rm -rf {idir}/events*')
  
writer = gezi.SummaryWriter(idir, False)

df = pd.read_csv(f'{idir}/metrics.csv')

steps = [x + 1 for x in range(len(df))]

factor = len(steps) / 10

def adjust(key):
  key = key.replace('metrics/auc', 'metrics/global_auc')
  key = key.replace('metrics/gauc2', 'metrics/auc')
  key = key.replace('metrics/auc/1', 'metrics/auc/old_user')
  key = key.replace('metrics/auc/2', 'metrics/auc/new_user')
  key = key.replace('metrics/global_auc/1', 'metrics/global_auc/old_doc')
  key = key.replace('metrics/global_auc/2', 'metrics/global_auc/new_doc')
  return key

keys =[x for x in df.columns if x != 'step']
for key in keys:
  if key in ['metrics/gauc', 'metrics/gauc3']:
    continue
  for val, step in zip(df[key].values, steps):
    if isinstance(val, str):
      continue
    if step % factor == 0:
      step = step / factor
      key_ = adjust(key)
      #print(key, key_)
      writer.scalar(key_, val, step)


