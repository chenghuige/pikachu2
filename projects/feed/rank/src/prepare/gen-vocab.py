#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-vocab.py
#        \author   chenghuige  
#          \date   2019-09-07 16:37:31.317958
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

# for prepare first you need to 
# cat ..feature_index_list | iconv -f gbk -t utf8 > feature_index 
# then using TextDataset you will get feat_field.txt  
# here we will modifiy them a bit to be able monitor embeddings as vocab.project
# save as feature.project and field.project

print('name\tid')
print('IGNORE\t0')
for line in sys.stdin:
 print('\t'.join(line.strip().split()))
