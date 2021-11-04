#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   filter-fields.py
#        \author   chenghuige  
#          \date   2020-03-14 11:40:06.637806
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

fields = []
for line in open(sys.argv[1]):
  field, count = line.strip().split()
  count = int(count)
  if count < 250:
    fields.append(field)

print(','.join(fields))
  
  
