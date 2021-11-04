#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   regex-fields.py
#        \author   chenghuige  
#          \date   2020-03-14 18:49:25.522353
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import re
import mmh3

pattern = sys.argv[1]

fields = []
fids = []
infos = []
for line in open(sys.argv[2]):
  l = line.strip().split()
  field, info = l[0], l[1]
  if re.search(pattern, field):
    fields.append(field)
    fids.append(mmh3.hash64(field)[0])
    infos.append(info)

print(','.join(fields))
#print(','.join(map(str, fids)))
print(list(zip(fields, infos)))
  
