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

train_uids = set()
uids = set()

for line in open('../train/behaviors.tsv'):
  uid = line.strip().split('\t')[1]
  train_uids.add(uid)

total = 0
new_uid_instances = 0
for line in open('./behaviors.tsv'):
  uid = line.strip().split('\t')[1]
  uids.add(uid)
  total += 1
  if uid not in train_uids:
    new_uid_instances += 1
  

print('num_uids:', len(uids))
print('new_uid_ratio:', len(uids - train_uids) / len(uids))
print('new_uid_insts_ratio:', new_uid_instances / total)

