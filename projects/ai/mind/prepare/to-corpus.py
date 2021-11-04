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
from tqdm import tqdm
from transformers import AutoTokenizer
import gezi

files = [
          './train/news.tsv', 
          './dev/news.tsv', 
          './test/news.tsv'
        ]

model_name = 'bert-base-cased'
model = f'/home/gezi/data/lm/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model)

vocab = gezi.Vocab(f'{model}/vocab.txt', fixed=True)

dids = set()
for file_ in files:
  total = len(open(file_).readlines())
  for line in tqdm(open(file_), total=total):
    l = line.strip().split('\t')
    did, title, abstract = l[0], l[3], l[4]
    if did in dids:
      continue
    dids.add(did)
    
    if abstract:
      text = title + ' ' + abstract
    else:
      text = title
    tokens = tokenizer.encode(text)
    tokens =  tokens[1:-1]
    print(' '.join(map(vocab.key, tokens)))
