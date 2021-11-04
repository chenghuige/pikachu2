#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   to-truth.py
#        \author   chenghuige  
#          \date   2020-08-22 02:23:32.058144
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

# this is for dev behavior
for line in open('../input/behaviors.tsv'):
  l = line.strip('\n').split('\t')
  id, history = l[0], l[-1]
  labels = []
  for item in history.split():
    labels.append(item.split('-')[-1])
  print(id, '[' + ','.join(labels) + ']', sep=' ')
