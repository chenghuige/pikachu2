#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   merge-index-field.py
#        \author   chenghuige  
#          \date   2019-09-16 11:49:36.127637
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags
FLAGS = flags.FLAGS

import sys 
import os
#import gezi
import mmh3
import six
from tqdm import tqdm

ENCODING = 'gb18030'

feat_file = sys.argv[1]

total = sum(1 for _ in open(feat_file))
for i, line in tqdm(enumerate(open(feat_file)), total=total):
  feat_name = line.strip().split('\t')[0]
  field_name = feat_name.split('\a')[0]
  if six.PY2:
    field_id = mmh3.hash64(field_name.decode('utf8').encode(ENCODING))[0]
    feat_id = mmh3.hash64(feat_name.decode('utf8').encode(ENCODING))[0] 
  else:
    field_id = mmh3.hash64(field_name.encode(ENCODING))[0]
    feat_id = mmh3.hash64(feat_name.encode(ENCODING))[0] 
  print(feat_name, feat_id, field_id, sep='\t')

