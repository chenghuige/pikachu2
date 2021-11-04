#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-doc.py
#        \author   chenghuige  
#          \date   2019-09-25 20:27:00.177678
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from KV import KV

doc_ids = ['20190923A0DYC400']
if len(sys.argv) > 1:
  doc_ids = sys.argv[1].split(',')

x = KV('110216', 'article_forward_index')
print(x.mget(doc_ids))
