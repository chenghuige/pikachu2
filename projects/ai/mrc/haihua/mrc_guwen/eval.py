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
  for id, label, pred in zip(ids, y_true, y_pred):
    preds.append(pred)
    labels.append(label)
    if len(preds) == 4:
      pred_idx = np.argmax(preds)
      label_idx = np.argmax(labels)
      preds = []
      labels = []
      qs += 1
      oks += int(pred_idx == label_idx)
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
      print(i, x['label'][i], predicts[i], file=f, sep=',')

def test_write(x, predicts, ofile, other=None):
  total = len(predicts)
  preds = []
  labels = ['A', 'B', 'C', 'D']
  with open(ofile, 'w') as f:
    print('id,label', file=f)
    for i in tqdm(range(total), desc='valid_write', leave=False):  
      id = x['id'][i]
      pred = predicts[i]
      preds.append(pred)
      label = None
      if len(preds) == 4:
        pred_idx = np.argmax(preds)
        label = labels[pred_idx]
        preds = []
        id = '%06d' % id
        print(f'{id},{label}', file=f)
