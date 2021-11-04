#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2020-05-24 12:17:53.108984
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
from tqdm import tqdm
import gezi
logging = gezi.logging 
import melt
from tqdm import tqdm
import random

from projects.ai.mango.src.util import *
from projects.ai.mango.src.config import *

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd

step = 1

def label_with_xor(lists):
    """
    >>> label_with_xor([1,1,1])
    False
    >>> label_with_xor([0,0,0])
    False
    >>> label_with_xor([0,])
    False
    >>> label_with_xor([1,])
    False
    >>> label_with_xor([0,1])
    True
    """
    if not lists:
        return False
    first = lists[0]
    for i in range(1, len(lists)):
        if lists[i] != first:
            return True
    return False

def group_auc(labels, preds, uids, selected_uids_list=[]):
    """Calculate group auc
    :param labels: list
    :param predict: list
    :param uids: list
    >>> gauc([1,1,0,0,1], [0, 0,1,0,1], ['a', 'a','a', 'b', 'b'])
    0.4
    >>> gauc([1,1,0,0,1], [1,1,0,0,1], ['a', 'a','a', 'b', 'b'])
    1.0
    >>> gauc([1,1,1,0,0], [1,1,0,0,1], ['a', 'a','a', 'b', 'b'])
    0.0
    >>> gauc([1,1,1,0,1], [1,1,0,0,1], ['a', 'a','a', 'b', 'b'])
    1.0
    """
    global step
    assert len(uids) == len(labels)
    assert len(uids) == len(preds)
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])

    for idx, truth in tqdm(enumerate(labels), desc='group', ascii=True, total=len(labels)):
        uid = uids[idx]
        group_score[uid].append(preds[idx])
        group_truth[uid].append(truth)

    total_auc = 0
    impression_total = 0
    total_aucs = [0] * len(selected_uids_list)
    impression_totals = [0] * len(selected_uids_list)
    for user_id in tqdm(group_truth, desc='gauc', ascii=True):
        if label_with_xor(group_truth[user_id]):
            auc = roc_auc_score(np.asarray(
                group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
            for i, selected_uids in enumerate(selected_uids_list):
              if user_id in selected_uids:
                total_aucs[i] += auc * len(group_truth[user_id])
                impression_totals[i] += len(group_truth[user_id])
    gauc = (float(total_auc) /
                 impression_total) if impression_total else 0
    gauc = round(gauc, 6)

    gaucs = [0.] * len(selected_uids_list)
    impression_rates = [0.] * len(selected_uids_list)
    for i in range(len(selected_uids_list)):
      gaucs[i] = (float(total_aucs[i]) /
                 impression_totals[i]) if impression_totals[i] else 0.
      gaucs[i] = round(gaucs[i], 6)

      impression_rates[i] = impression_totals[i] / impression_total

    return gauc, (gaucs, impression_rates)

def evaluate_df(df, prefix='eval/'):
  loss = log_loss(df.y_true.values, df.y_prob.values)
  auc = roc_auc_score(df.y_true.values, df.y_prob.values)
  
  if gezi.get('new_vids'):
    df_new_vids = df[df.vid.isin(gezi.get('new_vids'))]
    auc_new_vids = roc_auc_score(df_new_vids.y_true.values, df_new_vids.y_prob.values)
  else:
    auc_new_vids = 1.

  if 'watches' in df.columns:
    selected_uids_list = [
        set(df[df.watches==0].did.values), 
        set(df[(df.watches>0)&(df.watches<=28)].did.values), 
        set(df[(df.watches>28)&(df.watches<50)].did.values), 
        set(df[df.watches==50].did.values), 
      ]
    if gezi.get('new_uids'):
      selected_uids_list += [
        gezi.get('new_uids'), 
        set(df.did) - gezi.get('new_uids'),
      ]
  else:
    selected_uids_list = []

  gauc, (gaucs, impression_rates) = group_auc(df.y_true.values, df.y_prob.values, df.did.values, selected_uids_list=selected_uids_list)
  logging.debug('impression_rates', impression_rates)
  try:
    result = {
                'gauc/all': gauc, 
                'gauc/w0_uids': gaucs[0], # 0.0849
                'gauc/w1_uids': gaucs[1], # 1-28   0.325
                'gauc/w2_uids': gaucs[2], # 29-49  0.5765
                'gauc/w50_uids': gaucs[3], # 0.0136
                'gauc/new_uids': gaucs[4], # 0.33
                'gauc/old_uids': gaucs[5], # 0.66
                'loss': loss, 
                'auc/all': auc, 
                'auc/new_vids': auc_new_vids, 
              }
  except Exception:
    result = {
            'gauc/all': gauc, 
            'loss': loss, 
            'auc': auc, 
          }

  if prefix:
    result = gezi.dict_prefix(result, prefix)

  try:
    print(FLAGS.train_hour, '--------------------------gauc: [%.4f]' % gauc)
  except Exception:
    pass

  return result


def evaluate(y_true, y_pred, x, other):
  x['did'] = gezi.decode(x['did_'])
  x['vid'] = x['vid_']
  y_prob = other['prob']

  df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'index': x['index'], 
                     'vid': x['vid'], 'did': x['did'], 'watches': other['watches']})

  return evaluate_df(df)

if __name__ == '__main__':
  df = pd.read_csv(sys.argv[1])
 
  m = {'y_true': df.label, 'y_prob': df.score, 'did': df.did}
  if 'watches' in df.columns:
    m['watches'] = df.watches
  df_ = pd.DataFrame(m)
  res = evaluate_df(df_)
  print(res)
  gezi.pprint_df(res)