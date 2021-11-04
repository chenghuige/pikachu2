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

dir = sys.argv[1]
def main(argv):
  m = {}
  for line in open('%s/feat_fields.txt' % dir):
    field_name, field_id = line.strip().split('\t')
    m[field_name] = field_id
 
  for i, line in enumerate(open('%s/feature_index' % dir)):
    feat_name = line.strip().split('\t')[0]
    field_name = feat_name.split('\a')[0]
    field_id = m[field_name]
    feat_id = i + 1
    print(feat_name, feat_id, field_id, sep='\t')

if __name__ == '__main__':
  app.run(main)  
  
