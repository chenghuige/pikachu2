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
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix

from .config import *

def evaluate_df(df):
  gts = df.y_true.values
  probs = df.y_prob.values
  probs = [list(x) for x in probs]
  probs = np.asarray(probs)

  loss = log_loss(np.eye(NUM_CLASSES)[gts], probs)
  auc_yes = roc_auc_score([int(x) for x in gts == 0], [x[0] for x in probs])
  auc_no = roc_auc_score([int(x) for x in gts == 1], [x[1] for x in probs])
  
  preds = np.argmax(probs, axis=-1)
  tp = 0
  pred_pos = 0
  for gt, pred in zip(gts, preds):
    if pred not in {0, 1}:
      continue
    pred_pos += 1
    if gt == pred:
      tp += 1
  
  if pred_pos == 0:
    p = 0
  else:
    p = tp/pred_pos
  
  act_pos = np.sum((gts == 0).astype(int)) + np.sum((gts == 1).astype(int))
  r = tp / act_pos
  f1 = 0
  if p + r != 0:
    f1 = 2 * p *r / (p + r)

  cm = confusion_matrix(gts, preds)
  cm_args = dict(
          classes=CLASSES,
          normalize='true', 
          info='{}:{:.4f}'.format('f1', f1),
          title='',
          img_size=15,
    )
  cm_args['title'] = 'Recall'
  cm_ = gezi.plot.confusion_matrix(cm, **cm_args)
  wandb.log({'ConfusionMatrix/Recall': wandb.Image(cm_)})
  cm_args['title'] = 'Precision'
  cm_args['normalize'] = 'pred'
  cm_ = gezi.plot.confusion_matrix(cm, **cm_args)
  wandb.log({'ConfusionMatrix/Precision': wandb.Image(cm_)})
  cm_args['title'] = 'All'
  cm_args['normalize'] = 'all'
  cm_ = gezi.plot.confusion_matrix(cm / len(gts), **cm_args)
  wandb.log({'ConfusionMatrix/All': wandb.Image(cm_)})
  
  res = {
    'loss': loss,
    'auc/yes': auc_yes,
    'auc/no': auc_no,
    'precision/yesno': p,
    'recall/yesno': r,
    'f1/yesno': f1
  }
  res = gezi.dict_prefix(res, 'Metrics/')
  return res

def evaluate(y_true, y_pred, x):
  y_prob = gezi.softmax(y_pred)

  m = {
      'y_true': y_true, 
      'y_prob': list(y_prob), 
      'id': x['id'],
      }
  df = pd.DataFrame(m)

  return evaluate_df(df) 
