#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   check-conflict.py
#        \author   chenghuige  
#          \date   2019-10-26 14:00:08.584195
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
#import gezi
import mmh3
m = {}
for line in sys.stdin:
  field = line.rstrip().split('\t')[0].split('\a')[0]  
  #fid = gezi.hash_int64(field) % 10000
  #fid = gezi.hash_int64(field)
  fid = mmh3.hash64(field)[0] % 100000
  if fid not in m:  
    m[fid] = field 
  else:
    #if field != m[fid]:
    print('conflict:', field, m[fid], fid, line.strip())
   
print('num_field:', len(m))
  
