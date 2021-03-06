#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-folds.py
#        \author   chenghuige  
#          \date   2019-07-23 17:19:00.621274
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd   

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from config import *

df = pd.read_csv(f'{FLAGS.dir}/train.csv')

x = df['id_code']
y = df['diagnosis']
x, y = shuffle(x, y, random_state=RANDOM_STATE)

skf = StratifiedKFold(n_splits=NUM_FOLDS, random_state=RANDOM_STATE)

for i, (_, valid_index) in enumerate(skf.split(x, y)):
  x_valid, y_valid = x[valid_index], y[valid_index]
  df = pd.DataFrame()
  df['id_code'] = x_valid 
  df['diagnosis'] = y_valid 
  result_file = f'{FLAGS.dir}/train_{i}.csv'
  df.to_csv(result_file, index=False)
