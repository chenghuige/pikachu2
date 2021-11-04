#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-dict.py
#        \author   chenghuige  
#          \date   2020-08-21 14:12:10.754171
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from tqdm import tqdm
import gezi

uid_vocab = gezi.WordCounter()
did_vocab = gezi.WordCounter()

uid_vocab2 = gezi.WordCounter()
did_vocab2 = gezi.WordCounter()

  
files = [
  '../input/big/train/behaviors.tsv',
  '../input/big/dev/behaviors.tsv',
  '../input/big/test/behaviors.tsv'
  ]

for file in files:
  total = len(open(file).readlines())
  for line in tqdm(open(file), total=total):
    l = line.strip('\nt').split('\t')
    uid = l[1]
    uid_vocab.add(uid)
    if 'train' in file:
      uid_vocab2.add(uid)
    dids, dids2 = l[-2], l[-1]
    for did in dids.split():
      did_vocab.add(did)
      if 'train' in file:
        did_vocab2.add(did)
    for did in dids2.split():
      did = did.split('-')[0]
      did_vocab.add(did)
      if 'train' in file:
        did_vocab2.add(did)

uid_vocab.save('../input/big/uid.txt')
did_vocab.save('../input/big/did.txt')

uid_vocab2.save('../input/big/train/uid.txt')
did_vocab2.save('../input/big/train/did.txt')