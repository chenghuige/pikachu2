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

vocab_file = './entity.txt'
vocab = gezi.Vocab(vocab_file)
emb_height = vocab.size()

emb_size = len(open('./train/entity_embedding.vec').readline().strip().split()) - 1
print(emb_size)

emb = np.random.uniform(-0.05, 0.05,(emb_height, emb_size))
print(emb) 

emb = list(emb)

files = ['./train/entity_embedding.vec', './dev/entity_embedding.vec', './test/entity_embedding.vec']

entities = set()
for file_ in files:
  for line in tqdm(open(file_), total=emb_height):
    l = line.strip().split()
    entity, vals = l[0], l[1:]
    if entity in entities:
      continue
    entities.add(entity)
    vals = np.asarray(list(map(float, vals)))
    #vals = normalize(np.reshape(vals, (1,-1)))
    #vals /= np.sqrt(emb_size)
    vals = np.reshape(vals, (-1,))
    emb[vocab.id(entity)] = vals  

emb = np.asarray(emb)
print(emb)

#emb = normalize(emb)

np.save('./entity_emb.npy', emb)
