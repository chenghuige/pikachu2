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
from tempfile import NamedTemporaryFile

dir = sys.argv[1]

files = glob.glob('%s/tfrecord*' % dir)

def write_to_txt(data, file):
  # Hack for hdfs write
  out = NamedTemporaryFile('w')
  out.write('{}'.format(data))
  out.flush() 
  # os.system('rsync -a %s %s' %(out.name, file))
  os.system('scp %s %s' %(out.name, file))

num_records = 0 
for i, file in enumerate(files):
  try:
    num_records += int(file.split('.')[-1])
  except Exception:
    pass
  
print('num_records of', dir, num_records)
write_to_txt(num_records, '%s/num_records.txt' % dir)  
