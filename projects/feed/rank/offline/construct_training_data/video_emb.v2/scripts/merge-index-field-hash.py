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
from tqdm import tqdm

feat_file = sys.argv[1]

# TODO must be same as in gen-tfrecords.py
FEAT_DIM = 600000000
total = sum(1 for _ in open(feat_file))
for i, line in tqdm(enumerate(open(feat_file)), total=total):
  feat_name = line.strip().split('\t')[0]
  field_name = feat_name.split('\a')[0]
  #field_id = gezi.hash_int64(field_name) % FEAT_DIM 
  #feat_id = gezi.hash_int64(feat_name) % FEAT_DIM 
  field_id = mmh3.hash64(field_name)[0] % FEAT_DIM 
  feat_id = mmh3.hash64(feat_name)[0] % FEAT_DIM 
  print(feat_name, feat_id, field_id, sep='\t')

