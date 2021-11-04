#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   split-train-valid.py
#        \author   chenghuige  
#          \date   2019-10-13 15:45:55.358233
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob
import numpy as np

dir = sys.argv[1]
all_parts = int(sys.argv[2])
valid_parts = int(sys.argv[3])

files = glob.glob('%s/tfrecordss*' % dir)
assert len(files) == all_parts, files

np.random.shuffle(files)

train_parts = all_parts - valid_parts

valid_dir = None
train_dir = None
if valid_parts:
  valid_dir = '%s/valid' % dir
  command = 'mkdir -p %s' % valid_dir
  print(command)
  os.system(command)

if train_parts:
  train_dir = '%s/train' % dir
  command = 'mkdir -p %s' % train_dir
  print(command)
  os.system(command)

num_valids = 0
num_trains = 0

for i, file in enumerate(files):
  num_records = int(file.split('.')[-1])
  if i < valid_parts:
    num_valids += num_records
    command = 'mv %s %s' % (file, valid_dir)
    # print(command)
    os.system(command)
  else:
    num_trains += num_records
    command = 'mv %s %s' % (file, train_dir)
    # print(command)
    os.system(command)
  
print('num_valids', num_valids, 'num_trains', num_trains, 'valid_ratio', num_valids / (num_trains + num_valids))
if valid_dir:
  with open('%s/num_records.txt' % valid_dir, 'w') as out:
    print(num_valids, file=out)

if train_dir:
  with open('%s/num_records.txt' % train_dir, 'w') as out:
    print(num_trains, file=out)

  
