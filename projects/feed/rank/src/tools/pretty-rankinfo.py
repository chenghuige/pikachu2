#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pretty-rankinfo.py
#        \author   chenghuige  
#          \date   2019-12-03 12:00:55.432403
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from prettytable import PrettyTable

pt = PrettyTable(encoding='utf8')
from projects.feed.rank.src.infer.infer import head

names = head.split(',')
names = ['%d:%s' % (i + 1, names[i]) for i in range(len(names))]

if len(sys.argv) < 2:
  indexes = []
else:
  indexes = [int(x) for x in sys.argv[1].split(',')]
  names = [names[i] for i in indexes]

pt.field_names = names

for line in sys.stdin:
  l = line.rstrip('\n').split()
  if not indexes:
    indexes = range(len(l))
  #print '\t'.join(l[x] for x in indexes)
  pt.add_row([l[x] for x in indexes])

pt.align = 'l'
print(pt)

