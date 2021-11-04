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

flags.DEFINE_string('dir', '', '')

import sys 
import os
import six
import gezi

dir = sys.argv[1]

# TODO must be same as in gen-tfrecords.py
# FEAT_DIM = 600000000
ENCODING = 'gb18030'

for i, line in enumerate(open('%s/feature_index' % dir)):
  feat_name = line.strip().split('\t')[0]
  field_name = feat_name.split('\a')[0]
  if six.PY2:
    field_id = gezi.hash_int64(field_name.decode('utf8').encode(ENCODING)) % FEAT_DIM 
    feat_id = gezi.hash_int64(feat_name.decode('utf8').encode(ENCODING)) % FEAT_DIM 
  else:
    field_id = gezi.hash_int64(field_name.encode(ENCODING)) % FEAT_DIM 
    feat_id = gezi.hash_int64(feat_name.encode(ENCODING)) % FEAT_DIM 
  print(feat_name, feat_id, field_id, sep='\t')

if __name__ == '__main__':
  app.run(main)  
  
