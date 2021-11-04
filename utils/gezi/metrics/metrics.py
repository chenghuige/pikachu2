#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics.py
#        \author   chenghuige
#          \date   2019-08-17 08:17:00.550242
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

# https://github.com/qiaoguan/deep-ctr-prediction/blob/master/DeepCross/metric.py
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
# import multiprocessing
# from multiprocessing import Manager
# import pymp
from gezi import tqdm
from scipy.stats import weightedtau, kendalltau
from scipy.stats._stats import _kendall_dis
import math

import gezi

logging = gezi.logging

try:
  from gezi.metrics._stats import _weighted_inverse_sort
except Exception:
  pass

from numba import njit
from scipy.stats import rankdata


@njit
def _auc(actual, pred_ranks):
  n_pos = np.sum(actual)
  n_neg = len(actual) - n_pos
  return (np.sum(pred_ranks[actual == 1]) - n_pos *
          (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_auc(actual, predicted):
  # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
  pred_ranks = rankdata(predicted)
  return _auc(actual, pred_ranks)


def pr_auc(labels, preds):
  precision, recall, _thresholds = metrics.precision_recall_curve(labels, preds)
  area = metrics.auc(recall, precision)
  return area


def group_auc(labels, preds, uids, weighted=True):
  """Calculate group auc"""
  if len(uids) != len(labels):
    raise ValueError('{} {}'.format(len(uids), len(labels)))

  group_pred = defaultdict(lambda: [])
  group_truth = defaultdict(lambda: [])
  for idx, truth in tqdm(enumerate(labels),
                         total=len(labels),
                         ascii=True,
                         desc='make group',
                         leave=False):
    uid = uids[idx]
    score = preds[idx]
    truth = labels[idx]
    group_pred[uid].append(score)
    group_truth[uid].append(truth)

  group_flag = defaultdict(lambda: False)
  for uid in tqdm(set(uids),
                  total=len(set(uids)),
                  ascii=True,
                  desc='group flag',
                  leave=False):
    group_pred[uid] = np.array(group_pred[uid])
    group_truth[uid] = np.array(group_truth[uid])
    sum_truths = np.sum(group_truth[uid])
    group_flag[uid] = sum_truths > 0 and sum_truths < len(group_truth[uid])

  uids = [uid for uid in group_flag.keys() if group_flag[uid]]
  num_users = len(uids)
  # num instances 4544790 num users 811209 num users with auc 272392
  logging.debug('num instances', len(labels), 'num users', len(group_flag),
                'num users with auc', num_users)

  auc_scores = [None] * num_users
  impressions = [None] * num_users
  #auc_scores = Manager().list(auc_scores)
  #impressions = Manager().list(impressions)
  #num_cores = multiprocessing.cpu_count()

  #with pymp.Parallel(num_cores) as p:
  #  for i in tqdm(p.range(num_users)):
  for i in tqdm(range(num_users), total=num_users, ascii=True, desc='auc'):
    # auc_scores[i] = roc_auc_score(group_truth[uids[i]], group_pred[uids[i]])
    auc_scores[i] = fast_auc(group_truth[uids[i]], group_pred[uids[i]])
    impressions[i] = len(group_truth[uids[i]]) if weighted else 1
    auc_scores[i] *= impressions[i]

  total_auc = sum(auc_scores)
  total_impression = sum(impressions)
  # print('total_impression', total_impression)
  group_auc = float(total_auc) / total_impression
  return group_auc


def group_scores(labels,
                 preds,
                 uids,
                 selected_uids=None,
                 calc_auc=True,
                 corellation=True,
                 top_score=True,
                 min_click_duration=None,
                 topn=6,
                 weighted=False):
  """Calculate group auc"""
  if len(uids) != len(labels):
    raise ValueError('{} {}'.format(len(uids), len(labels)))

  if selected_uids:
    sunames, suids = zip(*(selected_uids.items()))
    num_suids = len(suids)
  else:
    suids = None
    num_suids = 0

  group_pred = defaultdict(lambda: [])
  group_truth = defaultdict(lambda: [])
  group_truth_binary = defaultdict(lambda: [])
  group_truth_binary2 = defaultdict(lambda: [])

  for idx, truth in tqdm(enumerate(labels),
                         total=len(labels),
                         ascii=True,
                         desc='make group',
                         leave=False):
    uid = uids[idx]
    score = preds[idx]
    truth = labels[idx]
    group_pred[uid].append(score)
    group_truth[uid].append(truth)

  group_flag = {}  # with click
  group_flag2 = {}  # with click and has valid duration
  group_flag_auc = {}
  for uid in tqdm(uids, ascii=True, desc='group flag', leave=False):
    group_pred[uid] = np.asarray(group_pred[uid])
    group_truth[uid] = np.asarray(group_truth[uid])
    if not min_click_duration:
      group_truth_binary[uid] = (group_truth[uid] != 0).astype(int)
      group_truth_binary2[uid] = (group_truth[uid] > 0).astype(int)
    else:
      group_truth_binary[
          uid] = group_truth[uid] >= min_click_duration or group_truth[uid] < 0
      group_truth_binary2[uid] = group_truth[uid] >= min_click_duration
    sum_truths = np.sum(group_truth_binary[uid])
    sum_truths2 = np.sum(group_truth_binary2[uid])
    group_flag[uid] = sum_truths > 0
    group_flag2[uid] = sum_truths2 > 0
    if calc_auc:
      group_flag_auc[uid] = sum_truths > 0 and sum_truths < len(
          group_truth_binary[uid])

  num_total_users = len(group_flag)
  logging.debug('num instances', len(labels), 'num users', num_total_users,
                'docs per user',
                len(labels) / num_total_users)

  if calc_auc:
    uids_auc = [uid for uid in group_flag_auc.keys() if group_flag_auc[uid]]
    num_users_auc = len(uids_auc)

    logging.debug('num users with auc', num_users_auc, 'auc user ratio',
                  num_users_auc / num_total_users)

    total_auc = 0.
    total_auc2 = 0.
    total_positive = 0
    total_impression = 0.

    total_aucs = [0.] * num_suids
    total_aucs2 = [0.] * num_suids
    total_positives = [0] * num_suids
    total_impressions = [0] * num_suids
    total_users = [0] * num_suids

    for i in tqdm(range(num_users_auc), ascii=True, desc='auc', leave=False):
      # auc
      truth = group_truth_binary[uids_auc[i]]
      pred = group_pred[uids_auc[i]]
      auc = roc_auc_score(truth, pred)
      impressions = len(truth)
      num_positive = sum(truth)

      total_auc += auc * impressions
      total_auc2 += auc
      total_positive += num_positive
      total_impression += impressions

      for j in range(num_suids):
        if uids_auc[i].split('\t')[0] in suids[j]:
          total_aucs[j] += auc * impressions
          total_aucs2[j] += auc
          total_positives[j] += num_positive
          total_impressions[j] += impressions
          total_users[j] += 1

    # auc
    group_auc = total_auc / total_impression
    group_auc2 = total_auc2 / num_users_auc
    pos_ratio = total_positive / total_impression

    group_aucs = [0.] * num_suids
    group_aucs2 = [0.] * num_suids
    pos_ratios = [0.] * num_suids
    impression_rates = [0.] * num_suids
    user_rates = [0.] * num_suids

    for j in range(num_suids):
      if total_impressions[j]:
        group_aucs[j] = total_aucs[j] / total_impressions[j]
        group_aucs2[j] = total_aucs2[j] / total_users[j]
        pos_ratios[j] = total_positives[j] / total_impressions[j]
      impression_rates[j] = total_impressions[j] / total_impression
      user_rates[j] = total_users[j] / num_users_auc

  if top_score:
    uids = [uid for uid in group_flag.keys() if group_flag[uid]]
    num_users = len(uids)

    top1_scores = [None] * num_users
    top1_clicks = [None] * num_users
    top3_clicks = [None] * num_users
    top3_impressions = [None] * num_users
    first_click_positions = [None] * num_users
    last_click_positions = [None] * num_users

    top1_scores_best = [None] * num_users
    top1_clicks_best = [None] * num_users
    top3_clicks_best = [None] * num_users

    # TODO maybe config 3,7,14..
    ndcgs_3 = [None] * num_users
    ndcgs_7 = [None] * num_users
    ndcgs_14 = [None] * num_users
    ndcgs = [None] * num_users

    for i in tqdm(range(num_users), ascii=True, desc='top click', leave=False):
      pred = group_pred[uids[i]]
      truth = group_truth_binary[uids[i]]
      index = np.argsort(-pred)
      top1_scores[i] = truth[index[0]]
      top1_clicks[i] = int(top1_scores[i] != 0)

      for j in range(len(index)):
        if truth[index[j]] != 0:
          first_click_positions[i] = j
          break
      for j in reversed(range(len(index))):
        if truth[index[j]] != 0:
          last_click_positions[i] = j
          break

      index_truth = np.argsort(-truth)

      scores = [truth[index[j]] for j in range(min(len(index), topn))]
      clicks = [int(x != 0) for x in scores]
      top3_clicks[i] = np.sum(clicks)
      top3_impressions[i] = len(scores)

      top1_scores_best[i] = truth[index_truth[0]]
      top1_clicks_best[i] = int(top1_scores_best[i] != 0)
      top3_clicks_best[i] = np.sum([
          int(truth[index_truth[j]] != 0)
          for j in range(min(len(index_truth), topn))
      ])

      best = truth[index_truth]
      r = truth[index]
      dcg_max_3 = gezi.dcg_at_k(best, 3)
      dcg_max_7 = gezi.dcg_at_k(best, 7)
      dcg_max_14 = gezi.dcg_at_k(best, 14)
      dcg_max = gezi.dcg_at_k(best, len(r))
      ndcgs_3[i] = gezi.dcg_at_k(r, 3) / dcg_max_3 if dcg_max_3 else 0.
      ndcgs_7[i] = gezi.dcg_at_k(r, 7) / dcg_max_7 if dcg_max_7 else 0.
      ndcgs_14[i] = gezi.dcg_at_k(r, 14) / dcg_max_14 if dcg_max_14 else 0.
      ndcgs[i] = gezi.dcg_at_k(r, len(r)) / dcg_max if dcg_max else 0.

    # top
    num_top3_impressions = np.sum(top3_impressions)
    top1_click = np.sum(top1_clicks) / num_users
    top3_click = np.sum(top3_clicks) / num_top3_impressions

    top1_click_best = np.sum(top1_clicks_best) / num_users
    top3_click_best = np.sum(top3_clicks_best) / num_top3_impressions

    first_click_position = np.sum(first_click_positions) / num_users
    last_click_position = np.sum(last_click_positions) / num_users

    ndcg_3 = np.sum(ndcgs_3) / num_users
    ndcg_7 = np.sum(ndcgs_7) / num_users
    ndcg_14 = np.sum(ndcgs_14) / num_users
    ndcg = np.sum(ndcgs) / num_users

    uids = [uid for uid in group_flag2.keys() if group_flag2[uid]]
    num_users = len(uids)

    top1_scores = [None] * num_users
    top3_scores = [None] * num_users
    top3_impressions = [None] * num_users

    top1_scores_best = [None] * num_users
    top3_scores_best = [None] * num_users

    ndcgs_3 = [None] * num_users
    ndcgs_7 = [None] * num_users
    ndcgs_14 = [None] * num_users
    ndcgs = [None] * num_users

    for i in tqdm(range(num_users), ascii=True, desc='top score', leave=False):
      pred = group_pred[uids[i]]
      truth = group_truth[uids[i]]
      flags = truth >= 0
      pred = pred[flags]
      truth = truth[flags]

      # top1 duration score
      index = np.argsort(-pred)
      top1_scores[i] = truth[index[0]]

      # top3 duration score
      scores = [truth[index[j]] for j in range(min(len(index), topn))]
      top3_scores[i] = np.sum(scores)
      top3_impressions[i] = len(scores)

      index_truth = np.argsort(-truth)
      top1_scores_best[i] = truth[index_truth[0]]
      top3_scores_best[i] = np.sum(
          [truth[index_truth[j]] for j in range(min(len(index_truth), topn))])

      best = truth[index_truth]
      r = truth[index]
      dcg_max_3 = gezi.dcg_at_k(best, 3)
      dcg_max_7 = gezi.dcg_at_k(best, 7)
      dcg_max_14 = gezi.dcg_at_k(best, 14)
      dcg_max = gezi.dcg_at_k(best, len(r))
      ndcgs_3[i] = gezi.dcg_at_k(r, 3) / dcg_max_3 if dcg_max_3 else 0.
      ndcgs_7[i] = gezi.dcg_at_k(r, 7) / dcg_max_7 if dcg_max_7 else 0.
      ndcgs_14[i] = gezi.dcg_at_k(r, 14) / dcg_max_14 if dcg_max_14 else 0.
      ndcgs[i] = gezi.dcg_at_k(r, len(r)) / dcg_max if dcg_max else 0.

    # top
    num_top3_impressions = np.sum(top3_impressions)
    top1_score = np.sum(top1_scores) / num_users
    top3_score = np.sum(top3_scores) / num_top3_impressions

    top1_score_best = np.sum(top1_scores_best) / num_users
    top3_score_best = np.sum(top3_scores_best) / num_top3_impressions

    ndcg_dur_3 = np.sum(ndcgs_3) / num_users
    ndcg_dur_7 = np.sum(ndcgs_7) / num_users
    ndcg_dur_14 = np.sum(ndcgs_14) / num_users
    ndcg_dur = np.sum(ndcgs) / num_users

    logging.debug('top1 score best:', top1_score_best, 'top3 score best:',
                  top3_score_best)

  concordant = np.nan
  weighted_concordant = np.nan
  if corellation:
    uids = [uid for uid in group_flag2.keys() if group_flag2[uid]]
    num_users = len(uids)
    logging.debug('num users with click', num_users, 'click users ratio',
                  num_users / num_total_users)

    inv_ratios = [None] * num_users
    weighted_invs = [None] * num_users
    # weighted_kendalls = [None] * num_users
    corellation_impressions = [None] * num_users

    for i in tqdm(range(num_users), ascii=True, desc='correlation',
                  leave=False):
      truth = group_truth[uids[i]]
      pred = group_pred[uids[i]]

      flags = truth >= 0
      pred = pred[flags]
      truth = truth[flags]

      corellation_impressions[i] = len(truth)

      inv_ratio = inverse_ratio(truth, pred)
      if weighted:
        weighted_inv = weighted_inverse(truth, pred)
      # weighted_kendall, _ = weightedtau(truth, pred)  # will core...

      inv_ratios[i] = inv_ratio * corellation_impressions[i]
      if weighted:
        weighted_invs[i] = weighted_inv * corellation_impressions[i]
      # weighted_kendalls[i] = weighted_kendall * corellation_impressions[i]

    filter_flag = np.asarray([not math.isnan(x) for x in inv_ratios])
    inv_ratios = np.asarray(inv_ratios)
    if weighted:
      weighted_invs = np.asarray(weighted_invs)
    # weighted_kendalls = np.asarray(weighted_kendalls)
    corellation_impressions = np.asarray(corellation_impressions)

    inv_ratios = inv_ratios[filter_flag]
    if weighted:
      weighted_invs = weighted_invs[filter_flag]
    # weighted_kendalls = weighted_kendalls[filter_flag]
    corellation_impressions = corellation_impressions[filter_flag]
    logging.debug('Num valid users for correlation',
                  len(corellation_impressions))

    total_impressions = np.sum(corellation_impressions)
    inv_ratio = np.sum(inv_ratios) / total_impressions
    if weighted:
      weighted_inv = np.sum(weighted_invs) / total_impressions
    # weighted_kendall = np.sum(weighted_kendalls) / total_impressions

  result = {}
  if calc_auc:
    result.update(dict(auc=group_auc, auc2=group_auc2, pos_ratio=pos_ratio))
    if suids:
      for i in range(len(suids)):
        uname = sunames[i]
        result[f'auc/{uname}'] = group_aucs[i]
        result[f'auc2/{uname}'] = group_aucs2[i]
        result[f'pos_ratio/{uname}'] = pos_ratios[i]
        result[f'impresssion_rate/{uname}'] = impression_rates[i]
        result[f'user_rate/{uname}'] = user_rates[i]

  if corellation:
    result['concordant'] = 1. - inv_ratio
    if weighted:
      result['weighted_concordant'] = 1. - weighted_inv
  if top_score:
    result.update(
        dict(
            top1_score=top1_score,
            top3_score=top3_score,
            top1_best=top1_score_best,
            top3_best=top3_score_best,
            top1_rate=(top1_score / top1_score_best),
            top3_rate=(top3_score / top3_score_best),
            top1_click=top1_click,
            top3_click=top3_click,
            top1_click_best=top1_click_best,
            top3_click_best=top3_click_best,
            top1_click_rate=(top1_click / top1_click_best),
            top3_click_rate=(top3_click / top3_click_best),
            first_click_position=first_click_position,
            last_click_position=last_click_position,
            ndcg3_click=ndcg_3,
            ndcg7_click=ndcg_7,
            ndcg14_click=ndcg_14,
            ndcg_click=ndcg,
            ndcg3_dur=ndcg_dur_3,
            ndcg7_dur=ndcg_dur_7,
            ndcg14_dur=ndcg_dur_14,
            ndcg_dur=ndcg_dur,
        ))

  return result


class CountInverse:

  def inverse_pairs(self, data):
    data = np.asarray(data)
    temp = np.zeros_like(data)
    return self.sort(data, 0, len(data) - 1, temp)

  def sort(self, data, left, right, temp):
    if right - left < 1:
      return 0

    if right - left == 1:
      if data[left] <= data[right]:
        return 0
      else:
        print('reverse', data[left], data[right])
        data[left], data[right] = data[right], data[left]
        return 1

    mid = (left + right) // 2
    dis = self.sort(data, left, mid, temp) + self.sort(data, mid + 1, right,
                                                       temp)

    # gezi.sprint(left)
    # gezi.sprint(right)
    # gezi.sprint(data)
    # gezi.sprint(temp)

    i = left
    j = mid + 1
    index = left

    while i <= mid and j <= right:
      if data[i] <= data[j]:
        temp[index] = data[i]
        i += 1
      else:
        # for k in range(i, mid + 1):
        #   print('reverse', data[k], data[j])
        temp[index] = data[j]
        dis += mid - i + 1
        j += 1
      index += 1

    while i <= mid:
      temp[index] = data[i]
      i += 1
      index += 1

    while j <= right:
      temp[index] = data[j]
      j += 1
      index += 1

    data, temp = temp, data

    return dis


class CountInverseWeightedBase:

  def inverse_pairs(self, data):
    weight_dis = np.asarray([-1] * len(data))
    data = np.asarray(data)
    temp = np.zeros_like(data)
    return self.sort(data, 0, len(data) - 1, temp, weight_dis)

  def sort(self, data, left, right, temp, weight_dis):
    if right - left < 1:
      return 0

    if right - left == 1:
      if data[left] <= data[right]:
        return 0
      else:
        data[left], data[right] = data[right], data[left]
        return data[right] - data[left]

    mid = (left + right) // 2
    dis_w = self.sort(data, left, mid, temp, weight_dis) + self.sort(
        data, mid + 1, right, temp, weight_dis)

    #print(f'111{data[left:mid + 1]},{data[mid + 1:right + 1]}--{temp[left:mid + 1]},{temp[mid + 1:right + 1]}')

    weight_dis[mid] = data[mid]
    for i in reversed(range(left, mid)):
      weight_dis[i] = weight_dis[i + 1] + data[i]

    i = left
    j = mid + 1
    index = left

    while i <= mid and j <= right:
      if data[i] <= data[j]:
        temp[index] = data[i]
        i += 1
      else:
        temp[index] = data[j]
        dis_w += weight_dis[i] - (mid - i + 1) * data[j]
        j += 1
      index += 1

    while i <= mid:
      temp[index] = data[i]
      i += 1
      index += 1

    while j <= right:
      temp[index] = data[j]
      j += 1
      index += 1

    #print(f'222{data[left:mid + 1]},{data[mid + 1:right + 1]}--{temp[left:mid + 1]},{temp[mid + 1:right + 1]}')

    i = left
    while i <= right:
      data[i] = temp[i]
      i += 1

    #print(f'333{data[left:mid + 1]},{data[mid + 1:right + 1]}--{temp[left:mid + 1]},{temp[mid + 1:right + 1]}')

    return dis_w


class CountInverseWeighted:

  def inverse_pairs(self, data):
    weight_dis = np.asarray([-1] * len(data))
    data = np.asarray(data)
    temp = np.zeros_like(data)
    return _weighted_inverse_sort(data, 0, len(data) - 1, temp, weight_dis)


def calc_inverse(x, y):
  #gezi.sprint(x)
  #gezi.sprint(y)
  index = np.argsort(y, kind='mergesort')
  #gezi.sprint(index)
  x = x[index]
  gezi.sprint(x)

  #print(l)

  calc = CountInverse()
  return calc.inverse_pairs(x)


def calc_inverse_weighted(x, y):
  #gezi.sprint(x)
  #gezi.sprint(y)
  index = np.argsort(y, kind='mergesort')
  #gezi.sprint(index)
  x = x[index]
  #gezi.sprint(x)

  #print(l)

  calc = CountInverseWeighted()
  return calc.inverse_pairs(x)


def weighted_inverse_base(x, y):
  index = np.argsort(y, kind='mergesort')
  x = x[index]
  calc = CountInverseWeightedBase()
  dis_w = calc.inverse_pairs(x)

  gezi.sprint(x)
  gezi.sprint(dis_w)

  n = len(x)
  tot = n * (n - 1) // 2

  weighted_inv = dis_w / tot

  return weighted_inv


def weighted_inverse(x, y):
  n = len(x)
  tot = n * (n - 1) // 2
  xtie = count_rank_tie(x)
  index = np.argsort(y, kind='mergesort')
  x = x[index]

  calc = CountInverseWeighted()
  dis_w = calc.inverse_pairs(x)

  #gezi.sprint(dis_w)

  tot -= xtie

  if tot == 0:
    return np.nan

  weighted_inv = dis_w / tot / np.mean(x[x != 0])

  return weighted_inv


def inverse_ratio_simple(x, y):
  xtie, ytie, con, dis = 0, 0, 0, 0
  n = len(x)
  for i in range(n - 1):
    for j in range(i + 1, n):
      test_tying_x = np.sign(x[i] - x[j])
      test_tying_y = np.sign(y[i] - y[j])
      judge = test_tying_x * test_tying_y
      if judge == 1:
        con += 1
      elif judge == -1:
        dis += 1

      if test_tying_y == 0 and test_tying_x != 0:
        ytie += 1
      elif test_tying_x == 0 and test_tying_y != 0:
        xtie += 1

  tau = (con - dis) / np.sqrt((con + dis + xtie) * (dis + con + ytie))

  inv_ratio = dis / (con + dis)

  return inv_ratio, tau


def weighted_inverse_simple(x, y):
  xtie, ytie, con, dis, con_w, dis_w = 0, 0, 0, 0, 0, 0
  n = len(x)
  for i in range(n - 1):
    for j in range(i + 1, n):
      diff = x[i] - x[j]
      weight = abs(diff)
      test_tying_x = np.sign(diff)
      test_tying_y = np.sign(y[i] - y[j])
      judge = test_tying_x * test_tying_y
      if judge == 1:
        con += 1
        con_w += weight
      elif judge == -1:
        dis += 1
        dis_w += weight

      if test_tying_y == 0 and test_tying_x != 0:
        ytie += 1
      elif test_tying_x == 0 and test_tying_y != 0:
        xtie += 1

  tau = (con - dis) / np.sqrt((con + dis + xtie) * (dis + con + ytie))

  inv_ratio = dis / (con + dis)
  weighted_inv_ratio = dis_w / (con_w + dis_w)

  print('con', con)
  print('dis', dis)
  print('con_w', con_w)
  print('dis_w', dis_w)

  return weighted_inv_ratio, inv_ratio, tau


def kendall_dis_simple(x, y):
  xtie, ytie, con, dis = 0, 0, 0, 0
  n = len(x)
  for i in range(n - 1):
    for j in range(i + 1, n):
      test_tying_x = np.sign(x[i] - x[j])
      test_tying_y = np.sign(y[i] - y[j])
      judge = test_tying_x * test_tying_y
      if judge == 1:
        con += 1
      elif judge == -1:
        dis += 1

      if test_tying_y == 0 and test_tying_x != 0:
        ytie += 1
      elif test_tying_x == 0 and test_tying_y != 0:
        xtie += 1

  return dis, con, xtie, ytie


def inverse_ratio(x, y):
  if isinstance(x, (list, tuple)):
    x = np.asarray(x)
  if isinstance(y, (list, tuple)):
    y = np.asarray(y)

  n = len(x)
  tot = n * (n - 1) // 2

  perm = np.argsort(y)  # sort on y and convert y to dense ranks
  x, y = x[perm], y[perm]
  y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

  # stable sort on x and convert x to dense ranks
  perm = np.argsort(x, kind='mergesort')
  x, y = x[perm], y[perm]
  x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

  dis = _kendall_dis(x, y)

  obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
  cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

  ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
  xtie = count_rank_tie(x)  # ties in x, stats
  ytie = count_rank_tie(y)  # ties in y, stats

  if xtie == tot or ytie == tot:
    return np.nan

  # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
  #               = con + dis + xtie + ytie - ntie
  # con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
  # tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)

  # Limit range to fix computational errors
  # tau = min(1., max(-1., tau))

  con = tot - dis - xtie - ytie + ntie

  inv_ratio = dis / (con + dis)

  if con + dis == 0:
    inv_ratio = np.nan
    # tau = np.nan

  # if math.isnan(tau):
  #   inv_ratio = np.nan
  # if math.isnan(inv_ratio):
  #   tau = np.nan

  return inv_ratio


inverse_rate = inverse_ratio


def inverse_ratio_click(labels, preds):
  filter_flag = labels > 0
  labels = labels[filter_flag]
  preds = preds[filter_flag]

  return inverse_ratio(labels, preds)


def kendall_dis(x, y):
  perm = np.argsort(y)  # sort on y and convert y to dense ranks
  x, y = x[perm], y[perm]
  y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

  # stable sort on x and convert x to dense ranks
  perm = np.argsort(x, kind='mergesort')
  x, y = x[perm], y[perm]
  x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

  dis = _kendall_dis(x, y)  # discordant pairs
  return dis


def count_rank_tie(ranks):
  cnt = np.bincount(ranks).astype('int64', copy=False)
  cnt = cnt[cnt > 1]
  return (cnt * (cnt - 1) // 2).sum()


if __name__ == '__main__':
  # import ptvsd

  # ptvsd.enable_attach()
  # ptvsd.break_into_debugger()

  # TODO kendall dis is 6 while inverse calc 7 not correct FIXME
  truth = np.asarray([3, 4, 5, 6, 0, 0])
  pred = np.asarray([7, 5, 4, 3, 1, 2])

  print(truth)
  print(pred)

  # print(kendalltau(pred, truth))
  # print(inverse_ratio(truth, pred))
  # print(inverse_ratio_simple(truth, pred))
  # print(kendall_dis(truth, pred))
  # print(kendall_dis(pred, truth))
  # print(kendall_dis_simple(truth, pred))
  # print(calc_inverse(truth, pred))
  #print(calc_inverse_weighted(truth, pred))
  print(weighted_inverse_simple(truth, pred))
  print(weighted_inverse_base(truth, pred))
  print(weighted_inverse(truth, pred))

  truth = np.asarray([30, 40, 50, 60, 0, 0])
  pred = np.asarray([7, 5, 4, 3, 10, 2])

  print(truth)
  print(pred)

  # print(kendalltau(pred, truth))
  # print(inverse_ratio(truth, pred))
  # print(inverse_ratio_simple(truth, pred))
  # print(kendall_dis(truth, pred))
  # print(kendall_dis(pred, truth))
  # print(kendall_dis_simple(truth, pred))
  # print(calc_inverse(truth, pred))
  #print(calc_inverse_weighted(truth, pred))
  print(weighted_inverse_simple(truth, pred))
  print(weighted_inverse_base(truth, pred))
  print(weighted_inverse(truth, pred))

  truth = np.asarray([3, 4, 5, 6, 1, 2])
  pred = np.asarray([6, 5, 4, 3, 0, 0])

  print(truth)
  print(pred)

  # print(kendalltau(pred, truth))
  # print(inverse_ratio(truth, pred))
  # print(inverse_ratio_simple(truth, pred))
  # print(kendall_dis(truth, pred))
  # print(kendall_dis(pred, truth))
  # print(kendall_dis_simple(truth, pred))
  # print(calc_inverse(truth, pred))
  #print(calc_inverse_weighted(truth, pred))
  print(weighted_inverse_simple(truth, pred))
  print(weighted_inverse_base(truth, pred))
  print(weighted_inverse(truth, pred))

  truth = np.asarray([3, 4, 5, 6, 1, 2])

  #pred = np.asarray([6, 5, 4, 3, 0, 0])
  #np.random.shuffle(pred)
  pred = np.asarray([3, 0, 5, 6, 0, 4])

  print(truth)
  print(pred)

  print(weighted_inverse_simple(truth, pred))
  print(calc_inverse(truth, pred))
  #print(weighted_inverse_base(truth, pred))
  #print(weighted_inverse(truth, pred))

  exit(0)

  pred = np.asarray(
      [1.2, 4.5, 5.4, 0.5, 0.1, 0.01, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26])
  truth = np.asarray([3, 4, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  print(kendalltau(pred, truth))
  print(kendalltau(truth, pred))
  print(inverse_ratio(truth, pred))

  print(kendall_dis(truth, pred))
  print(kendall_dis(pred, truth))

  pred = np.asarray([1.2, 4.5, 5.4, 0.5, 0.1, 0.01, 0.2])
  truth = np.asarray([3, 4, 5, 1, 0.2, 0.01, 0.1])

  print(kendalltau(pred, truth))
  print(kendalltau(truth, pred))
  print(inverse_ratio(truth, pred))
  print(weighted_inverse_base(truth, pred))
  print(weighted_inverse_simple(truth, pred))

  print(kendall_dis(truth, pred))
  print(kendall_dis(pred, truth))

  pred = np.asarray([0.1, 0.01, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26])
  truth = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0])

  print(kendalltau(pred, truth))
  print(kendalltau(truth, pred))
  print(inverse_ratio(truth, pred))
  print(kendall_dis(truth, pred))
  print(kendall_dis(pred, truth))
