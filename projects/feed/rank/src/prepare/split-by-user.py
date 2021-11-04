#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   split-by-user.py
#        \author   chenghuige  
#          \date   2019-08-19 09:54:20.684307
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
from tqdm import tqdm

total = 37000000

user_infos = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
  os.system('mkdir -p %s' % out_dir)

out_train = open(os.path.join(out_dir, 'train.txt'), 'w')
out_valid = open(os.path.join(out_dir, 'valid.txt'), 'w')

users = [line.rstrip().split('\t')[0] for line in open(user_infos)]
users = np.asarray(users)
np.random.shuffle(users)

num_users = len(users)
num_valid_users = int(len(users) / 8)

valid_users = set(users[:num_valid_users])
train_users = set(users[num_valid_users:])

for line in tqdm(sys.stdin, total=total, ascii=True):
  uid = line.rstrip().split('\t')[2] 
  if uid in valid_users:
    print(line, end='', file=out_valid)
  else: 
    print(line, end='', file=out_train)
