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

model_name = 'bert-base-cased'
model = f'/home/gezi/data/lm/{model_name}'
vocab_file = f'{model}/vocab.txt'
vocab = gezi.Vocab(vocab_file, fixed=True)
emb_height = vocab.size()

print(vocab.id('i'))

emb_size = len(open('./vectors.txt').readline().strip().split()) - 1
print(emb_size)

emb = np.random.uniform(-0.05, 0.05,(emb_height, emb_size))
print(emb) 

emb = list(emb)

for line in tqdm(open('./vectors.txt'), total=emb_height):
  l = line.strip().split()
  word, vals = l[0], l[1:]
  vals = np.asarray(list(map(float, vals)))
  #vals = normalize(np.reshape(vals, (1,-1)))
  #vals /= np.sqrt(emb_size)
  vals = np.reshape(vals, (-1,))
  emb[vocab.id(word)] = vals  

emb = np.asarray(emb)
print(emb)

#emb = normalize(emb)

np.save('./emb.npy', emb)
