#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   clean-other.py
#        \author   chenghuige  
#          \date   2020-04-19 21:39:18.751530
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd

df_train_clean = pd.read_csv('./train_preprocessed.csv')
df_test_clean = pd.read_csv('./test_preprocessed.csv')

df_clean = pd.concat([df_train_clean, df_test_clean])

df = pd.read_csv('../jigsaw-toxic-comment-train.csv')
df = df.sort_values(['id'])
df_clean = df_clean[df_clean.id.isin(df.id.values)]
df_clean = df_clean.sort_values(['id'])

df['comment_text'] = df_clean['comment_text'].values
  
df.to_csv('./jigsaw-toxic-comment-train-clean-other.csv', index=False)

