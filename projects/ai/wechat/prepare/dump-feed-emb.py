#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dump-feed-emb.py
#        \author   chenghuige  
#          \date   2021-06-12 04:27:41.641287
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import pandas as pd
import gezi
from gezi import tqdm
from sklearn.preprocessing import normalize
np.random.seed(1024)

vocab = gezi.Vocab(f'../input/doc_vocab.txt')
emb_height = vocab.size()
feed_file = '../input/feed_embeddings.csv'
emb_size = len(open(feed_file).readlines()[1].strip().split(',')[1].split()) 
print(emb_height, emb_size)

emb = np.zeros((/emb_height, emb_size))
# emb = np.random.uniform(-0.05, 0.05,(emb_height, emb_size))

df = pd.read_csv(feed_file)
for row in tqdm(df.itertuples(), total=len(df), ascii=False, desc='feed_embeddings'):
  row = row._asdict()
  assert vocab.id(int(row['feedid'])) > 1, row
  emb[vocab.id(int(row['feedid']))] = np.asarray(list(map(float, row['feed_embedding'].split())))

print(np.min(emb), np.max(emb), np.mean(emb))
print(np.min(emb[0]), np.max(emb[0]), np.mean(emb[0]))
print(np.min(emb[1]), np.max(emb[1]), np.mean(emb[1]))
print(np.min(emb[2]), np.max(emb[2]), np.mean(emb[2]))
#l = []
#for i, row in enumerate(emb):
#  print(i, np.mean(row))

np.save('../input/feed_embeddings.npy', emb)
emb = normalize(emb)
np.save('../input/feed_norm_embeddings.npy', emb)

