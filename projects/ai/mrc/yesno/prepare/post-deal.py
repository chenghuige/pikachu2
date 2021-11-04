#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   post-deal.py
#        \author   chenghuige  
#          \date   2021-01-10 15:45:08.907496
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd

d_train = pd.read_csv('../input/train.csv')
d_train['query_in_train'] = True
d_train['id'] = list(range(len(d_train)))

d_dev = pd.read_csv('../input/dev.csv')

queries_train = set(d_train['query'])

d_dev['query_in_train'] = d_dev['query'].apply(lambda x: x in queries_train)
d_dev['id'] = list(range(len(d_dev)))

d_dev.to_csv('../input/dev.csv', index=False)
d_train.to_csv('../input/train.csv', index=False)
