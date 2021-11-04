#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   stats-new-user.py
#        \author   chenghuige  
#          \date   2020-08-23 00:41:05.173388
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

train_dids = set()
train_dids2 = set()
dids = set()
dids2 = set()

for line in open('../train/behaviors.tsv'):
  l = line.strip().split('\t')
  history = l[2]
  for did in history.split():
    train_dids2.add(did)
  impressions = l[-1]
  for item in impressions.split():
    did, _ = item.split('-') 
    train_dids.add(did)
    train_dids2.add(did)

total = 0
new_did_insts = 0
total_impres = 0
new_did_impres = 0
for line in open('./behaviors.tsv'):
  l = line.strip().split('\t')
  history = l[2]
  for did in history.split():
    dids2.add(did)
  impressions = l[-1]
  find = 0
  for item in impressions.split():
    did = item.split('-')[0]
    dids.add(did)
    dids2.add(did)
    if did not in train_dids:
      find += 1
    total += 1
  new_did_insts += find
  if find:
    new_did_impres += 1
  total_impres += 1

print('num_dids:', len(dids), 'num_dids2:', len(dids2))
print('new_did_ratio:', len(dids - train_dids) / len(dids), 'new_did_ratio2:', len(dids - train_dids2) / len(dids))
print('new_did_insts_ratio:', new_did_insts / total)
print('new_did_impress_ratio:', new_did_impres / total_impres)

