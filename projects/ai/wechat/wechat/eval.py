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
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

import os
import gezi
from gezi import logging

from absl import app
import wandb
import json
import numpy as np
import pandas as pd
import pymp
from multiprocessing import Pool, Manager, cpu_count
import pymp
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import collections
from collections import OrderedDict, Counter, defaultdict
from icecream import ic

from numba import njit
from scipy.stats import rankdata
import glob
import functools

from gezi import logging, tqdm
from wechat.config import *
from wechat.util import *
import wechat.fast_uauc as fuauc
from wechat.ensemble import *

@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

dic = {}

def uAUC__(index, df, names):
  index_ = 0
  if isinstance(df, list):
    # print([len(df[i]) for i in range(len(df))])
    index_ = int(index / len(FLAGS.action_list))
    df = df[index_]
    index = index % len(FLAGS.action_list)
  
  action = FLAGS.action_list[index]
  key = f'{names[index_]}/{action}'
  dic['uauc'][key] = uAUC(df[action].values, preds=df[f'{action}_pred'].values, user_id_list=df['uid'].astype(str).tolist())

def uAUC_(index, labels, preds, user_id_list):
  return uAUC(labels, preds, user_id_list, index)

def uAUC(labels, preds, user_id_list, index=None):
    """Calculate user AUC"""

    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in tqdm(enumerate(labels), ascii=False, desc='uAUC', total=len(labels), leave=False):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    if index is not None:
      start, end = gezi.get_fold(len(user_id_list), FLAGS.auc_threads, index)
      user_id_list = user_id_list[start: end]

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = fast_auc(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    
    user_auc = float(total_auc)/size if size > 0 else 0.5

    if index is not None:
      dic['total_auc'][index] = total_auc
      dic['num_users'][index] = size

    return user_auc

def compute_weighted_score(score_dict, weight_dict):
    '''基于多个行为的uAUC值，计算加权uAUC
    Input:
        scores_dict: 多个行为的uAUC值映射字典, dict
        weights_dict: 多个行为的权重映射字典, dict
    Output:
        score: 加权uAUC值, float
    '''
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict.get(action, 0.))
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score

def evaluate_df(df, is_last=True):
  dnames = ['all'] 
  dfs = [df]
  
  if (is_last or FLAGS.always_eval_all) and (not FLAGS.simple_eval):
    if not FLAGS.eval_exclnonfirst:
      if 'is_first' in df.columns:
        dnames += ['first']
        dfs += [df[df.is_first == 1]]
    if 'num_poss' in df.columns:
      dnames += ['cold', 'hot', ]
      dfs += [df[df.num_poss == 0], df[df.num_poss > 0]]
    if 'fresh' in df.columns:
      dnames += ['colddoc', 'hotdoc']
      dfs += [df[df.fresh == 0], df[df.fresh > 0]]
    
    if FLAGS.eval_ab_users:
      # 模拟测试test_a和test_b 
      dnames += ['test_a', 'test_b']
      dfs += [df[df.uid % 2 == 0], df[df.uid % 2 == 1]]
      dnames += ['test_b_cold', 'test_b_hot']
      dfs += [df[(df.uid % 2 == 1) & (df.num_poss == 0)], df[(df.uid % 2 == 1) & (df.num_poss > 0)]]
      dnames += ['test_b_colddoc', 'test_b_hotdoc']
      dfs += [df[(df.uid % 2 == 1) & (df.fresh == 0)], df[(df.uid % 2 == 1) & (df.fresh > 0)]]

  if FLAGS.work_mode != 'train':
    ic(list(zip(dnames, [len(df) for df in dfs], [len(set(df.uid)) for df in dfs], 
                [len(set(df.uid)) / len(set(dfs[0].uid)) for df in dfs])))
  
  results = OrderedDict()
  loss_results = OrderedDict()
  if FLAGS.nw == 1 or FLAGS.work_mode == 'train':
    # 一般这个比较安全 单进程 异步调用时间也还好 大概1分半 8个df
    weights = [weights_map[action] for action in FLAGS.action_list]
    t = tqdm(zip(dnames, dfs), total=len(dfs), desc='uauc', ascii=False, leave=False)
    for dname, df in t:
      logging.debug(dname, len(df), len(set(df.uid)))
      t.set_postfix({'df_name': dname})
      trues, preds = [], []
      loss = 0.
      loss_list = []
      for i, action in enumerate(FLAGS.action_list):
        trues.append(df[action])
        preds.append(df[f'{action}_pred'])
        loss_ = log_loss(df[action].values, df[f'{action}_pred'].values)
        loss += weights[i] * loss_
        loss_list.append(loss_)
        loss_results[f'loss_{dname}/{action}'] = loss_

      trues = np.stack(trues, axis=-1)
      preds = np.stack(preds, axis=-1)
      key = 'score' if not dname else f'{dname}/score'
      score, uaucs = fuauc.uAUC(trues, preds, df.uid.values, weights)
      results[key] = score

      for uauc, action in zip(uaucs, FLAGS.action_list):
       results[f'{dname}/{action}'] = uauc
      
      if sum(weights):
        loss /= sum(weights)
      key = 'loss' if not dname else f'{dname}/loss'
      results[key] = loss
      loss_results[f'loss_{dname}/wmean'] = loss
      loss_results[f'loss_{dname}/mean'] = np.mean(loss_list)
    results.update(loss_results)
  else:
    # TODO 一般没有问题 --mode=valid 但是--mode=train 在v100 机器总是带来问题core p40机器似乎没出过问题
    # Failed setting context: CUDA_ERROR_NOT_INITIALIZED: initialization
    works = len(FLAGS.action_list)
    works *= len(dfs)
    partial_func = functools.partial(uAUC__, df=dfs, names=dnames)
    dic['uauc'] = Manager().dict()
  #   nw = max(min(FLAGS.nw or int(cpu_count() / 2 - 1), works), 1)
    nw = max(min(FLAGS.nw or cpu_count(), works), 1)
    if nw > 1:
      # 额 在A100 机器 同步验证 模式 训练过程中 有很小的概率还是会hang
      with pymp.Parallel(nw) as p:
        for i in tqdm(p.range(works), desc='uaucs', leave=False):
          partial_func(i)
    else:
      for i in tqdm(range(works), desc='uaucs', leave=False):
        partial_func(i)

    score_detail = dic['uauc']
    for dname, df in zip(dnames, dfs):
      logging.debug(dname, len(df), len(set(df.uid)))
      score = 0.0
      weights_sum = 0.0
      loss = 0.
      loss_list = []
      for action in FLAGS.action_list:
        weight = weights_map.get(action, 0.)
        action_ = f'{dname}/{action}'
        score += weight * score_detail[action_]
        loss_ = log_loss(df[action].values, df[f'{action}_pred'].values)
        loss += weight * loss_
        loss_list.append(loss_)
        loss_results[f'loss_{dname}/{action}'] = loss_
        weights_sum += weight

      if weights_sum > 0:
        score /= weights_sum  
        loss /= weights_sum
      key = 'score' if not dname else f'{dname}/score'
      results[key] = score 
      for action in FLAGS.action_list:
        action_ = f'{dname}/{action}'
        results[action_] = score_detail[action_]
      key = 'loss' if not dname else f'{dname}/loss'
      results[key] = loss
      loss_results[f'loss_{dname}/wmean'] = loss
      loss_results[f'loss_{dname}/mean'] = np.mean(loss_list)

    # results.update(score_detail)
    results.update(loss_results)
      
  for key in results:
    if 'loss' in key:
      results[key] = 1. - results[key]

  results = gezi.dict_prefix(results, 'Metrics/')
  return results

def evaluate(y_true, y_pred, x, is_last=True, from_logits=True):
  # 目前在tione v100测试遇到问题就是train + valid 再最后结束tf 要释放gup资源 这时候要async_eval
  # 或者eval FLAGS.nw != 1 多进程 Multiprocess 必然冲突 目前使用全异步验证async_valid来避免 另外就是可以注释始终FLAGS.nw == 1
  # 复现注意第一valid没问题 第二次core sh run/2/small-13.sh --gpus=2 --bspg --lr_scale --mts --rv=.small --vie=0.5 --async_valid=0
  # 或者 --nw=1 --async_eval 及时eval单进程 但是异步eval 启动一个进程做eval 也会core类似
  if is_last and FLAGS.nw == 1 and FLAGS.work_mode != 'train':
    # if last eval use more cpu resource to speedup
    FLAGS.nw = 0
    
  if from_logits:
    y_prob = gezi.sigmoid(y_pred)
  else:
    y_prob = y_pred

  m = {
        'uid': x['userid'], 
        'did': x['feedid'],
       }
  
  for key in eval_other_keys:
    if key in x:
      m[key] = x[key]
      
  if len(y_prob.shape) == 1:
    y_prob = np.expand_dims(y_prob, -1)
  
  action_list = FLAGS.action_list
  true_list = []
  for i, action in enumerate(action_list):
    m[action] = x[action]
    true_list.append(x[action])
    if i >= y_prob.shape[1] or FLAGS.tower_train:
      m[f'{action}_pred'] = y_prob[:, 0]
    else:
      m[f'{action}_pred'] = y_prob[:, i]

  if 'finish' in m:
    m['finish'] = (m['finish'] > 0.99).astype(int)
  if 'stay' in m:
    m['stay'] = (m['stay'] > 0.5).astype(int)
          
  df = pd.DataFrame(m)

  if FLAGS.eval_exclnonfirst:
    df = df[df.is_first == 1]

  if FLAGS.work_mode != 'train':
    ic('eval len', len(df), 'users', len(set(df.uid)))

  return evaluate_df(df, is_last=is_last)

def valid_write(ids, label, predicts, ofile):
  write_result(ids, predicts, ofile, is_infer=False)

def infer_write(ids, predicts, ofile):
  write_result(ids, predicts, ofile, is_infer=True)

def write_result(ids, predicts, ofile, is_infer=True):
  x = ids
  uid = x['userid']
  did = x['feedid']
  prob = gezi.sigmoid(predicts)
  m = {
      'userid': uid,
      'feedid': did,
  }
  if len(prob.shape) == 1:
    prob = np.expand_dims(prob, -1)

  action_list = [x for x in FLAGS.loss_list]
  for item in FLAGS.action_list:
    if item not in action_list:
      action_list.append(item)

  for i, action in enumerate(action_list):
    if i < prob.shape[-1]:
      m[action] = prob[:, i]
    else:
      m[action] = prob[:, 0]
  
  df = pd.DataFrame(m)[['userid', 'feedid'] + FLAGS.action_list]
  df.to_csv(ofile, index=False)

def main(_):
  FLAGS.work_mode = 'valid'
  valid_file = '../input/valid.csv'
  with gezi.Timer('read_valid', print_fn=ic):
    df_truth = pd.read_csv('../input/valid.csv')
  df_truth = df_truth.sort_values(['userid', 'feedid'])
  files = FLAGS.ensemble_files if not FLAGS.ensemble_pattern else glob.glob(FLAGS.ensemble_pattern)
  if not files[0].endswith('.csv'):
    files = [f'../working/offline/{FLAGS.ensemble_version}/{x}/valid.csv' for x in files]
  weights = [float(x) for x in FLAGS.ensemble_weights]
  if not weights:
    weights = [1.] * len(files)

  ic(len(files), list(zip(files, weights)))

#  dfs = Manager().list()
#   ps = min(len(files), cpu_count())
#   with pymp.Parallel(ps) as p:
#     for i in tqdm(p.range(len(files))):
#       df_pred = pd.read_csv(files[i])
#       df_pred = df_pred.sort_values(['userid', 'feedid'])
#       dfs.append(df_pred)
  
  dfs = []
  for i in tqdm(range(len(files))):
    df_pred = pd.read_csv(files[i])
    df_pred = df_pred.sort_values(['userid', 'feedid'])
    dfs.append(df_pred)

  with gezi.Timer('ensemble', print_fn=ic):
    df_pred = ensemble(dfs, weights)

  preds = []
  FLAGS.action_list = FLAGS.action_list or ACTION_LIST
  for action in FLAGS.action_list:
    preds.append(df_pred[action].values)
    
  preds = np.stack(preds, 1)
  x = {}
  for col in df_truth.columns:
    x[col] = df_truth[col].values
  res = evaluate(None, preds, x, from_logits=False)
  res = gezi.dict_rename(res, 'Metrics/', '')
  gezi.pprint_dict(res)

  if len(files) > 1:
    path_ = os.path.dirname(files[-1])
    mdir = os.path.dirname(path_)
    mname = os.path.basename(path_).split('-')[0]
    odir = f'{mdir}/{FLAGS.ensemble_dir}'
    gezi.try_mkdir(odir)
    with open(f'{odir}/models.txt', 'w') as f:
      for i, (file, weight) in enumerate(zip(files, weights)):
        print(i, file, weight, file=f)
    if FLAGS.write_ensemble_valid:
      ofile = f'{odir}/valid.csv'
      ic(ofile)
      df_pred.to_csv(ofile, index=False)
    writer = gezi.DfWriter(odir, filename='metrics.csv')
  else:
    odir = os.path.dirname(files[-1])
    if os.path.exists(f'{odir}/metrics.csv'):
      gezi.try_remove(f'{odir}/metrics.csv.bak')
      os.rename(f'{odir}/metrics.csv', f'{odir}/metrics.csv.bak')
    writer = gezi.DfWriter(odir, filename='metrics.csv')

  writer.write({}, mode='w')
  writer.write(res)


if __name__ == '__main__':
  app.run(main)  
