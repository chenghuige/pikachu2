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
    fname, fid = line.strip().split('\t')
    m[fname] = fid
 
  for line in open('%s/feature_index' % dir):
    fname = line.strip().split('\t')[0].split('\a')[0]
    fid = m[fname]
    print(line.strip(), fid, sep='\t')

if __name__ == '__main__':
  app.run(main)  
  
