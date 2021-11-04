#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   all-clicks.py
#        \author   chenghuige  
#          \date   2020-06-12 20:36:11.292515
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
from tqdm import tqdm
import json
import time
import gezi

dcs = []
for k in tqdm(range(30)):
  d = gezi.read_parquet(f'../input/train/part_{k + 1}/context.parquet')
  d = d[d.label==1][['did', 'vid', 'timestamp']]
  d['day'] = k + 1
  dcs += [d]
dc = pd.concat(dcs)

dc = dc.sort_values(['did', 'timestamp'], ascending=[True, False])

dc.to_csv('../input/all/clicks.csv', index=False)
