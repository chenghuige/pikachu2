#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   split-train-valid.py
#        \author   chenghuige  
#          \date   2019-10-13 15:45:55.358233
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob

dir = sys.argv[1]

files = glob.glob('%s/tfrecord*' % dir)

num_records = 0 
for i, file in enumerate(files):
  try:
    num_records += int(file.split('.')[-1])
  except Exception:
    if os.path.exists(file):
      command = 'rm -rf %s' % file
    os.system(command)
    pass
  
print('num_records', num_records)
with open('%s/num_records.txt' % dir, 'w') as out:
  print(num_records, file=out)


  
