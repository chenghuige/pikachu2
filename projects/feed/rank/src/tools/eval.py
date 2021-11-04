#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2019-08-23 06:22:27.324981
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np 
from sklearn.metrics import roc_auc_score, log_loss
import random
import pandas as pd
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('version', None, '')
flags.DEFINE_integer('level', 20, '20 print info, 21 will not print info, 5 will print debug')

import gezi

logging = gezi.logging


def main(_):
  logging.init('/tmp', file='eval.html', level=FLAGS.level)
  
  y = []
  durations = []
  uids = []
  y_ = []

  num_bads = 0
  for line in sys.stdin:
    uid, duration, pred = line.rstrip().split()
      
    duration = int(duration)

    pred = float(pred)
    if pred < 0:
      num_bads += 1
      continue
    
    if duration > 60 * 60 * 12: 
      duration = 60
    
    y.append(int(duration > 0))
    durations.append(duration)
    uids.append(uid)
    y_.append(pred)

  logging.info('num_pos exampes:', sum(y), 'num_examples:', len(y), 'pos ratio:', sum(y) / len(y), 'num_bads:', num_bads, 'bads_ratio:', num_bads / len(y))

  y = np.asarray(y)
  y_ = np.asarray(y_)
  durations = np.asarray(durations)
  uids = np.asarray(uids)

  logits = y_
  # y_ = gezi.sigmoid(y_)

  auc = roc_auc_score(y, y_)
  loss = log_loss(y, y_)

  logging.info('auc', auc, 'loss', loss)

  inv_ratio = gezi.metrics.inverse_ratio(durations, logits)

  click_flag = durations > 0
  durations_click = durations[click_flag]
  logits_click = logits[click_flag]
  uids_click = uids[click_flag]

  inv_ratio_click = gezi.metrics.inverse_ratio(durations[click_flag], logits[click_flag])  

  logging.info('time_auc', 1. - inv_ratio, 'click/time_auc', 1 - inv_ratio_click)

  weighted_inv = gezi.metrics.weighted_inverse(durations, logits)  
  logging.info('weighted_time_auc', 1. - weighted_inv)

  weighted_inv_click = gezi.metrics.weighted_inverse(durations_click, logits_click)  
  logging.info('click/weighted_time_auc', 1. - weighted_inv_click)

  results = {'pos_ratio': sum(y) / len(y),
             'auc': auc, 
             'time_auc': 1. - inv_ratio,
             'weighted_time_auc':1. - weighted_inv,
             'click/time_auc': 1. - inv_ratio_click,
             'click/weighted_time_auc': 1. - weighted_inv_click}

  group_results = \
    gezi.metrics.group_scores(durations, y_, uids)
    
  click_results = \
    gezi.metrics.group_scores(durations_click, logits_click, uids_click, auc=False)  

  group_results = gezi.dict_prefix(group_results, 'group/')
  group_results = gezi.dict_rename(group_results, 'concordant', 'time_auc')

  click_results = gezi.dict_prefix(click_results, 'group/click/')
  click_results = gezi.dict_rename(click_results, 'concordant', 'time_auc')

  results.update(group_results)
  results.update(click_results)
  # print(results, file=sys.stderr)
  results['gold/auc'] = (results['group/auc'] * results['group/click/time_auc']) ** 0.5

  global_info = {}
  global_info['click_ratio'] = np.sum(y) / len(y)
  global_info['time_per_show'] = np.sum(durations) / len(durations)
  global_info['time_per_click'] = np.sum(durations) / len(durations_click)
  global_info['time_per_user'] = np.sum(durations) / len(set(uids))
  
  results.update(global_info)

  print('valid_metrics:{}'.format(['%s:%.5f' % (name, val) for name, val in results.items() if not isinstance(val, str)] + ['version:{}'.format(FLAGS.version)]))

  #gezi.pprint_dict(results, print_fn=logging.info)

  importants = ['gold/auc', 'group/auc', 'group/click/time_auc', \
                'auc', 'time_auc', 'weighted_time_auc', 'click/time_auc', 'click/weighted_time_auc', \
                'group/time_auc', 'group/weighted_time_auc', 'group/top3_click_rate', \
                'group/click/weighted_time_auc', 'group/click/top1_rate']

  # for key in importants:
  #   print(key, results[key])

  def rename(key):
    return key.replace('weighted_time', 'wtime') \
              .replace('version', 'v') \
              .replace('group', 'g') \
              .replace('click', 'c')
    
  gezi.dict_del(results, 'group/click/auc')
  gezi.dict_del(results, 'group/click/pos_ratio')
  gezi.pprint_dict(results, importants, rename_fn=rename, print_fn=logging.info)

if __name__ == '__main__':
  app.run(main)
  
