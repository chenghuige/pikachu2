#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-vid-vocab.py
#        \author   chenghuige  
#          \date   2021-10-14 14:00:03.650372
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import pandas as pd
import gezi

valid_vocab = gezi.Vocab(num_reserved_ids=0)
for line in open('../input/pairwise/label.tsv'):
  q, c, _ = line.strip().split()
  valid_vocab.add(q)
  valid_vocab.add(c)
valid_vocab.save('../input/pairwise/vid.vocab')

test_vocab = gezi.Vocab(num_reserved_ids=0)
d = pd.read_csv('../input/test/ids.csv')
for vid in d['id'].values:
  test_vocab.add(vid)
test_vocab.save('../input/test/vid.vocab')

