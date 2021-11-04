#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-uid-vocab.py
#        \author   chenghuige  
#          \date   2021-06-10 11:34:25.066478
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os

import gezi
from gezi import tqdm

counter = gezi.WordCounter()

files = ['user_action.csv', 'test_a.csv', 'feed_embeddings.csv']
counts = [1, 0, 0]
for i, file_ in enumerate(files):
  is_first = True
  file = f'../input/{file_}'
  total = gezi.get_num_lines(file)
  for line in tqdm(open(file), total=total, desc=f'{file_}-doc'):
    if is_first:
      is_first = False
      continue
    idx = 1 if i != 2 else 0
    doc = line.rstrip().split(',')[idx]
    
    counter.add(doc, counts[i])

counter.save('../input/doc_vocab.txt')
vocab = gezi.Vocab(f'../input/doc_vocab.txt')
print(vocab.size(), vocab.size(min_count=5))
