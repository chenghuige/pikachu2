#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pd-mean.py
#        \author   chenghuige  
#          \date   2020-04-21 14:02:19.183349
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import glob 
import numpy as np

if os.path.isdir(sys.argv[1]):
  idir = sys.argv[1]
  files = glob.glob(f'{idir}/valid_*.csv')
else:
  files = sys.argv[1:]
  idir = './'

dfs = [pd.read_csv(file_) for file_ in files]

#dfs[1].pred *= 1.4
#dfs[0].pred *= 0.6 

df = pd.concat(dfs)

key = 'id' if 'id' in df.columns else 'index'
df = df.groupby([key], as_index=False).mean()
df = df.sort_values(key)

dfs[0] = dfs[0].sort_values(key)

df['did'] = dfs[0].did.values

df = df.to_csv(f'{idir}/ensemble.csv', index=False)
  
