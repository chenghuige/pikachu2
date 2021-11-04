#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige
#          \date   2021-01-09 17:51:06.853603
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gezi

import wandb
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import collections
from collections import OrderedDict, Counter, defaultdict
from gezi import logging, tqdm
from .config import *

def evaluate(y_true, y_pred, x, other):
  ids = x['id']
  qs = 0
  oks = 0
  y_pred = y_pred.astype(np.float32)
  ids, y_true, y_pred = list(ids), list(y_true), list(y_pred)

  for id, label, pred in zip(ids, y_true, y_pred):
    pred_idx = np.argmax(pred)
    qs += 1
    oks += int(pred_idx == label)
  acc = oks / qs
  res = {'acc': acc}
  res = gezi.dict_prefix(res, 'Metrics/')
  return res

# --write_valid 控制写valid result, predicts对应model(input)的前向输出 other是一个dict 里面的key 对应model.out_keys
def valid_write(x, labels, predicts, ofile, other=None):
  total = len(labels)
  with open(ofile, 'w') as f:
    print('id,label,pred', file=f)
    for i in tqdm(range(total), desc='valid_write', leave=False):  
      print(i, labels[i], predicts[i], file=f, sep=',')

def test_write(x, predicts, ofile, other=None):
  total = len(predicts)
  predicts = predicts.astype(np.float32)
  labels = ['A', 'B', 'C', 'D']
  with open(ofile, 'w') as f:
    print('id,label', file=f)
    for i in tqdm(range(total), desc='test_write', leave=False):  
      id = x['id'][i]
      id_str = '%06d' % id
      preds = predicts[i]
      pred_idx = np.argmax(preds)
      label = labels[pred_idx]
      print(f'{id_str},{label}', file=f)
