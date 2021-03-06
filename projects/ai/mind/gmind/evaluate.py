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
import gezi
logging = gezi.logging 
import melt

import random
import json

from gmind.util import *
from gmind.config import *

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd

from gezi import tqdm

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

def dcg_score(y_true, y_score, k=10):
  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k])
  gains = 2 ** y_true - 1
  discounts = np.log2(np.arange(len(y_true)) + 2)
  return np.sum(gains / discounts)
    
def ndcg_score(y_true, y_score, k=10):
  best = dcg_score(y_true, y_true, k)
  actual = dcg_score(y_true, y_score, k)
  return actual / best

def mrr_score(y_true, y_score):
  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order)
  rr_score = y_true / (np.arange(len(y_true)) + 1)
  return np.sum(rr_score) / np.sum(y_true)

def group_auc(labels, preds, uids, selected_uids_list=[], adjust_ratio=0.151048):
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

  for idx, truth in tqdm(enumerate(labels), desc='group', ascii=True, total=len(labels), leave=True):
    uid = uids[idx]
    group_score[uid].append(preds[idx])
    group_truth[uid].append(truth)

  total_auc, total_auc2 = 0, 0
  total_mrr, total_ndcg5, total_ndcg10 = 0, 0, 0
  impression_total = 0
  total_aucs = [0] * len(selected_uids_list)
  total_aucs2 = [0] * len(selected_uids_list)
  impression_totals = [0] * len(selected_uids_list)
  for user_id in tqdm(group_truth, desc='gauc', ascii=True, leave=True):
    if label_with_xor(group_truth[user_id]):
      y_true = np.asarray(group_truth[user_id])
      y_score = np.asarray(group_score[user_id])
      auc = roc_auc_score(y_true, y_score)
      mrr = mrr_score(y_true,y_score)
      ndcg5 = ndcg_score(y_true,y_score,5)
      ndcg10 = ndcg_score(y_true,y_score,10)
      total_auc += auc * len(group_truth[user_id])
      total_auc2 += auc
      total_mrr += mrr
      total_ndcg5 += ndcg5
      total_ndcg10 += ndcg10
      impression_total += len(group_truth[user_id])
      for i, selected_uids in enumerate(selected_uids_list):
        if user_id in selected_uids:
          total_aucs[i] += auc * len(group_truth[user_id])
          total_aucs2[i] += auc
          impression_totals[i] += len(group_truth[user_id])
  gauc = (float(total_auc) /
                impression_total) if impression_total else 0

  num_uids = len(group_truth)
  gauc2 = float(total_auc2) / num_uids

  mrr = float(total_mrr) / num_uids
  ndcg5 = float(total_ndcg5) / num_uids
  ndcg10 = float(total_ndcg10) / num_uids

  gaucs = [0.] * len(selected_uids_list)
  gaucs2 = [0.] * len(selected_uids_list)
  for i in range(len(selected_uids_list)):
    gaucs[i] =  float(total_aucs[i]) / impression_totals[i]
    gaucs2[i] = float(total_aucs2[i]) / len(selected_uids_list[i])
  gauc3 = gaucs2[0] * (1 - adjust_ratio) + gaucs2[1] * adjust_ratio

  result = {
    'gauc2': gauc2,
  }
  for i in range(len(selected_uids_list)):
    result[f'gauc2/{i + 1}'] = gaucs2[i]

  result.update({
    'gauc3': gauc3,
    'gauc': gauc,
    'mrr': mrr,
    'ndcg5': ndcg5,
    'ndcg10': ndcg10
  })

  return result

def adjust(key):
  key = key.replace('Metrics/auc', 'Metrics/global_auc')
  key = key.replace('Metrics/gauc2', 'Metrics/auc')
  key = key.replace('Metrics/gauc3', 'Metrics/auc/adjust')
  key = key.replace('Metrics/auc/1', 'Metrics/auc/old_user')
  key = key.replace('Metrics/auc/2', 'Metrics/auc/new_user')
  key = key.replace('Metrics/global_auc/1', 'Metrics/global_auc/old_doc')
  key = key.replace('Metrics/global_auc/2', 'Metrics/global_auc/new_doc')
  return key

def evaluate_df(df):
  loss = log_loss(df.y_true.values, df.y_prob.values)
  auc = roc_auc_score(df.y_true.values, df.y_prob.values)
  
  df1 = df[df.did_in_train==1]
  df2 = df[df.did_in_train==0] 
  logging.debug('did in train ratio', len(df1) / len(df), len(df2) / len(df))
  auc_1 = roc_auc_score(df1.y_true.values, df1.y_prob.values)
  auc_2 = roc_auc_score(df2.y_true.values, df2.y_prob.values)

  df1 = df[df.uid_in_train==1]
  df2 = df[df.uid_in_train==0]
  logging.debug('uid insts in train ratio', len(df1) / len(df), len(df2) / len(df))
  logging.debug('uid impresses in train ratio', len(set(df1.impression_id)) / len(set(df.impression_id)), len(set(df2.impression_id)) / len(set(df.impression_id)))
  result = group_auc(df.y_true.values, df.y_prob.values, df.impression_id.values, selected_uids_list=[set(df1.impression_id), set(df2.impression_id)])
  
  result['auc/all'] = auc
  result['auc/1'] = auc_1
  result['auc/2'] = auc_2
  
  result['loss'] = loss

  result_ = gezi.dict_prefix(result, 'Metrics/')
  
  result = {}
  for key in result_:
    result[adjust(key)] = result_[key]

  return result

def evaluate(y_true, y_pred, x):
  y_prob = gezi.sigmoid(y_pred)
  df = pd.DataFrame({
                      'y_true': y_true,
                      'y_prob': y_prob, 'impression_id': x['impression_id'], 
                      'uid': x['uid'], 'did': x['did'], 
                      'history_len': x['hist_len'], 
                      'uid_in_train': x['uid_in_train'], 'did_in_train': x['did_in_train']
                     })

  return evaluate_df(df)

if __name__ == '__main__':
  print('------loading truth')
  df_truth = pd.read_csv('../input/dev/truth.txt', header=None, sep=' ', names=['impression_id', 'y_trues'])
  df_truth.y_trues = df_truth.y_trues.apply(lambda x: json.loads(x))
  impression_ids, truths = [], []

  print('------loading valid')
  df = pd.read_csv(sys.argv[1], header=None, sep=' ', names=['impression_id', 'y_probs'])
  df.y_probs = df.y_probs.apply(lambda x: json.loads(x))
  # df.y_probs = df.y_probs.apply(lambda x: json.loads(x))
  impression_ids2, probs = [], []

  for _, row in tqdm(df.iterrows(), total=len(df), ascii=True, leave=False):
    for y_prob in row['y_probs']:
      impression_ids2.append(row['impression_id'])
      probs.append(y_prob)

  for _, row in tqdm(df_truth.iterrows(), total=len(df), ascii=True, leave=False):
    for y_true in row['y_trues']:
      impression_ids.append(row['impression_id'])
      truths.append(y_true)

  assert len(truths) == len(probs), f'{len(truths)} != {len(probs)}'

  for i in range(len(impression_ids)):
    assert impression_ids[i] == impression_ids2[i], i

  m = {'y_true': truths, 'y_prob': probs, 'impression_id': impression_ids}
  print('dict to df')
  df_ = pd.DataFrame(m)
  res = evaluate_df(df_)
  gezi.pprint_df(res)
