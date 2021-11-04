#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   clean.py
#        \author   chenghuige  
#          \date   2020-04-19 13:06:04.396228
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count

import tokenizer
import gezi

ifile = sys.argv[1]
# d = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
df = pd.read_csv(ifile)

m = Manager().dict()

def deal(index):
  total_df = len(df)
  start, end = gezi.get_fold(total_df, cpu_count(), index)
  total = end - start
  df_ = df.iloc[start:end]
  l = []
  for _, row in tqdm(df_.iterrows(), total=total):
    text = tokenizer.clean(row['comment_text'])
    l.append(text)
  m[index] = l
 
with Pool(cpu_count()) as p:
  p.map(deal, range(cpu_count()))

ofile = ifile[:-4] + '-clean.csv'

df['comment_text'] = np.concatenate([m[i] for i in range(cpu_count())])

df.to_csv(ofile, index=False)

