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

keys = ['key', 'tag', 'word', 'char', 'author', 'singer', 'song']

for key in keys:
  file = f'../input/{key}_corpus.txt'
  counter = gezi.WordCounter()
  total = gezi.get_num_lines(file)
  for line in tqdm(open(file), total=total, desc=key):
    for item in line.strip().split():
      counter.add(item)
  counter.save(f'../input/{key}_vocab.txt')
  vocab = gezi.Vocab(f'../input/{key}_vocab.txt')
  print(vocab.size(), vocab.size(min_count=5))

