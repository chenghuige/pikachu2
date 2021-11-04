#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   merge-emb.py
#        \author   chenghuige  
#          \date   2020-06-03 22:36:34.375581
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
import gezi
from sklearn.preprocessing import normalize
from tqdm import tqdm

np.random.seed(1024)

UNK_ID = 1

vocab = gezi.Vocab(f'../input/{sys.argv[3]}_vocab.txt')
emb_height = vocab.size()
emb_size = len(open(sys.argv[1]).readline().strip().split()) - 1
print(emb_size)

# emb = np.random.uniform(-0.05, 0.05,(emb_height, emb_size))
emb = np.zeros((emb_height, emb_size))

print(np.min(emb), np.max(emb), np.mean(emb))
print(np.min(emb[0]), np.max(emb[0]), np.mean(emb[0]))
print(np.min(emb[1]), np.max(emb[1]), np.mean(emb[1]))

emb = list(emb)

num_lines = gezi.get_num_lines(sys.argv[1])
for line in tqdm(open(sys.argv[1]), total=num_lines, desc=sys.argv[1]):
  l = line.strip().split()
  word, vals = l[0], l[1:]
  vals = np.asarray(list(map(float, vals)))
  vals = np.reshape(vals, (-1,))
  try:
    emb[vocab.id(word)] = vals  
  except Exception:
    print(word)
    emb[UNK_ID] = vals

emb = np.asarray(emb)
print(emb)

if len(sys.argv) > 4 and sys.argv[4] == 'norm':
  emb = normalize(emb)

print(np.min(emb), np.max(emb), np.mean(emb))
print(np.min(emb[0]), np.max(emb[0]), np.mean(emb[0]))
print(np.min(emb[1]), np.max(emb[1]), np.mean(emb[1]))

np.save(sys.argv[2], emb)
