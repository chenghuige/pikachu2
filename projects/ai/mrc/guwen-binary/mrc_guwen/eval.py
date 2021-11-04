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
  preds = []
  labels = []
  qs = 0
  oks = 0
  id_ = None
  ids, y_true, y_pred = list(ids), list(y_true), list(y_pred)
  ids.append(None)
  y_true.append(None)
  y_pred.append(None)
  for id, label, pred in zip(ids, y_true, y_pred):
    if preds and (id != id_):
      pred_idx = np.argmax(preds)
      label_idx = np.argmax(labels)
      preds = []
      labels = []
      qs += 1
      oks += int(pred_idx == label_idx)
    preds.append(pred)
    labels.append(label)
    id_ = id
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
  preds = []
  labels = ['A', 'B', 'C', 'D']
  id_ = None
  with open(ofile, 'w') as f:
    print('id,label', file=f)
    for i in tqdm(range(total + 1), desc='valid_write', leave=False):  
      if i != total:
        id = x['id'][i]
        pred = predicts[i]
      else:
        id, pred = None, None
      label = None
      if preds and (id != id_):
        pred_idx = np.argmax(preds)
        label = labels[pred_idx]
        preds = []
        id_str = '%06d' % id_
        print(f'{id_str},{label}', file=f)
      preds.append(pred)
      id_ = id
