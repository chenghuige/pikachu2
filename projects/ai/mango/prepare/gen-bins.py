#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-bins.py
#        \author   chenghuige  
#          \date   2020-06-12 19:18:30.966556
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import pandas as pd
from tqdm import tqdm
import gezi

dis = []
for i in tqdm(range(30)):
  dis += [gezi.read_parquet(f'../input/train/part_{i + 1}/item.parquet')]
dis += [gezi.read_parquet('../input/eval/item.parquet')]
di = pd.concat(dis)
  
k = 10
w =[ i/k for i in range(k+1)]
w =di.describe (percentiles=w)[4:4+k+1]

w.to_csv('../input/all/bins.csv', index=False)

