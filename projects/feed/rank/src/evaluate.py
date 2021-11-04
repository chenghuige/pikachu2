#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2019-07-28 08:43:41.067128
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import partial
import traceback
import torch

import gezi
from gezi.summary import SummaryWriter
import melt
logging = gezi.logging
from projects.feed.rank.src import util 

dataset = None 

# TODO add user group auc
def evaluate(y, y_):
  y_ = gezi.sigmoid(y_)
  auc = roc_auc_score(y, y_)
  loss = log_loss(y, y_)
  return ['auc', 'loss'], [auc, loss]

def evaluate2(y, y_, ids):
  y_ = gezi.sigmoid(y_)
  auc = roc_auc_score(y, y_)
  loss = log_loss(y, y_)
  gauc = gezi.metrics.group_auc(y, y_, ids)
  return ['auc', 'gauc', 'loss'], [auc, gauc, loss]

# TODO score here means duration may be also rename score to dur
metrics = [
            'gold/auc', 
            'group/auc', 'group/auc2', 
            # 'auc', 'clickmids/auc', 'cold/auc', 
            'group/click/time_auc', 'click/rmse',
            'group/top1_click', 'group/click/top1_dur',
            'group/ndcg3_click', 'group/click/ndcg3_dur', 
            'loss/click', 'loss/dur'
          ]
importants = ['info'] + metrics
keys = metrics

def rename(key):
  return key.replace('time_auc', 'tauc') \
            .replace('version', 'v') \
            .replace('group', 'g') \
            .replace('click', 'c') \
            .replace('quality', 'q')

baseline_logger = None
logger = None

# uids means group id can also be impresssion time
def eval(durations, pred, uids, selected_uids=None, group=False, mode='all', 
         video_times=None, page_times=None, read_completion_rates=None,
         max_duration=None, df=None):
  # 2 means dur related as dur can has -1 unknown wihch is a click, we will evaluate treat it as click for auc 
  # or other click metrics but filter for duration related metrics like time_auc
  y_true = (durations != 0).astype(np.int32) 

  pos_ratio = sum(y_true) / len(y_true)
  prob = pred
  result = {}
  # logits = pred
  # prob = gezi.sigmoid(logits)

  try:
    auc = roc_auc_score(y_true, prob)
  except Exception:
    auc = np.nan

  try:
    loss = log_loss(y_true, prob)
  except Exception:
    loss = np.nan

  # TODO dur prob calc should use same code as loss.py
  dur_flags = durations >= 0
  durations2 = durations[dur_flags]
  dur_prob_true = np.minimum(durations / float(max_duration or FLAGS.max_duration), 1.)
  dur_prob_true2 = dur_prob_true[dur_flags]
  prob2 = prob[dur_flags]
  prob2 = np.minimum(prob2, 1.) # online ori lr score might has 1.+ 
  try:
    loss_dur = torch.nn.BCELoss()(torch.as_tensor(prob2, dtype=torch.float), torch.as_tensor(dur_prob_true2, dtype=torch.float)).numpy()
  except Exception:
    logging.warning(traceback.format_exc())
    loss_dur = np.nan
  logging.debug('auc', auc, 'loss', loss, 'loss_dur', loss_dur)
  
  # timer = gezi.Timer('inverse_ratio', True)
  inv_ratio = gezi.metrics.inverse_ratio(durations2, prob2)
  mse, mae = mean_squared_error(dur_prob_true2, prob2), mean_absolute_error(dur_prob_true2, prob2)
   
  result = dict(auc=auc, 
                mse=mse, 
                mae=mae, 
                concordant=(1. - inv_ratio), 
                mean=np.mean(prob))
  result.update({'loss/click': loss, 'loss/dur': loss_dur, 'stats/pos_ratio': pos_ratio})

  if video_times is not None:
    try:
      vtime_stats = {}
      inv_ratio_vtime = gezi.metrics.inverse_ratio(video_times, prob)
      # if one video is viewed by more then max_duration(600s) then finish ratio is also 1
      # finish_ratios = np.minimum(durations / np.maximum(np.minimum(video_times, FLAGS.max_duration), FLAGS.min_video_time), 1.)
      finish_ratios = np.minimum(durations / np.minimum(video_times, FLAGS.max_duration), 1.)
      inv_ratio_finish = gezi.metrics.inverse_ratio(finish_ratios, prob)
      vtime_stats = dict(vtime_auc=1. - inv_ratio_vtime,
                        finish_auc=1. - inv_ratio_finish)
      result.update(vtime_stats)
    except Exception:
      pass

  if read_completion_rates is not None:
    rcr_flags = read_completion_rates >= 0.
    inv_ratio_rcr = gezi.metrics.inverse_ratio(read_completion_rates[rcr_flags], prob[rcr_flags])
    result['rcr_auc'] = 1. - inv_ratio_rcr

  if page_times is not None:
    inv_ratio_ptime = gezi.metrics.inverse_ratio(page_times, prob)
    result['ptime_auc'] = 1. - inv_ratio_ptime
  
  click_flag = durations != 0
  click_flag2 = durations > 0
  durations_click = durations[click_flag]
  durations_click2 = durations[click_flag2]
  prob_click = prob[click_flag]
  prob_click2 = prob[click_flag2]
  assert len(durations_click) > 0, 'all druations 0?'
  if len(durations_click) == 0:
    logging.warning('all durations 0')
    result = gezi.dict_rename(result, 'concordant', 'time_auc')
    return result

  if mode == 'all':
    stats_result = {}
    if df is not None:
      stats_result['mean_histories'] = df.num_histories.mean()
      stats_result['mean_tw_histories'] = df.num_tw_histories.mean()
      stats_result['mean_vd_histories'] = df.num_vd_histories.mean()
    stats_result['num_instances'] = len(y_true)
    stats_result['click_ratio'] = np.sum(y_true) / len(y_true)
    total_durs = np.sum(durations2) / 60.
    stats_result['time_per_show'] = total_durs / len(durations2) 
    stats_result['time_per_click'] = total_durs / 60. / len(durations_click2)
    stats_result['time_per_user'] = total_durs / 60. / len(set(uids))
    if video_times is not None:
      total_vtimes = np.sum(video_times) / 60.
      total_click_vtimes = np.sum(video_times[click_flag]) / 60.
      stats_result['vtime_per_show'] = total_vtimes / len(durations2) 
      stats_result['vtime_per_click'] = total_click_vtimes / len(durations_click2)
      stats_result['vtime_per_user'] = total_click_vtimes / len(set(uids))
    stats_result = gezi.dict_prefix(stats_result, 'stats/')
    result.update(stats_result)

  inv_ratio_click = gezi.metrics.inverse_ratio(durations_click2, prob_click2)  

  dur_prob_true_click2 = dur_prob_true[click_flag2]
  try:
    loss_dur_click = torch.nn.BCELoss()(torch.as_tensor(prob_click2, dtype=torch.float), torch.as_tensor(dur_prob_true_click2, dtype=torch.float)).numpy()
  except Exception:
    loss_dur_click = np.nan
  mse_click, mae_click = mean_squared_error(dur_prob_true_click2, prob_click2), mean_absolute_error(dur_prob_true_click2, prob_click2)

  result['click/concordant'] = (1. - inv_ratio_click)
  result['click/loss/dur'] = loss_dur_click 
  result['click/mse'] = mse_click
  result['click/rmse'] = 1. - mse_click ** 0.5
  result['click/mae'] = mae_click
  result['click/mean'] = np.mean(prob_click)

  if not group: 
    result = gezi.dict_rename(result, 'concordant', 'time_auc')
    return result
  
  if mode != 'dur':
    group_result = gezi.metrics.group_scores(durations, pred, uids, selected_uids)
    group_result = gezi.dict_prefix(group_result, 'group/')
    result.update(group_result)
  
  if mode != 'click':
    uids_click = uids[click_flag]
    group_click_result = gezi.metrics.group_scores(durations_click, prob_click, uids_click, selected_uids, calc_auc=False)
    group_click_result = gezi.dict_prefix(group_click_result, 'group/click/')
    result.update(group_click_result)

  result = gezi.dict_rename(result, 'concordant', 'time_auc')
  result = gezi.dict_rename(result, 'top1_rate', 'top1_dur_rate')
  result = gezi.dict_rename(result, 'top3_rate', 'top3_dur_rate')
  result = gezi.dict_rename(result, 'top1_score', 'top1_dur')
  result = gezi.dict_rename(result, 'top3_score', 'top3_dur')
  result = gezi.dict_rename(result, 'top1_best', 'top1_dur_best')
  result = gezi.dict_rename(result, 'top3_best', 'top3_dur_best')

  gezi.dict_del(result, 'group/click/auc')
  gezi.dict_del(result, 'group/click/pos_ratio')

  try:
    result['gold/auc'] = (result['group/auc'] * result['group/click/time_auc']) ** 0.5
  except Exception:
    pass

  return result  

def parse_base_file():
  file = f'{FLAGS.base_result_dir}/{FLAGS.valid_hour}/{FLAGS.eval_product}_metrics_offline.csv' 
  if FLAGS.eval_group_by_impression:
    file = file.replace('.csv', '_impression.csv')
  if os.path.exists(file):
    df = pd.read_csv(file)
    df = df[df.abtest==45600]
    m = dict([(x, df[x].values[0]) for x in df.columns])
    return m
  else:
    logging.debug(f'Not find base file {file} will calc online result')
  return {}

online_result = {}
is_first = True
step = 0

def evaluate_rank_(df, selected_uids={}, group=False, mode='all', key='pred', group_by_impression=False):  
  if not group_by_impression:
    group_key = 'mid'
  else:
    group_key = 'impression'
  video_times = df.video_time.values if 'video_time' in df.columns else None
  page_times = df.page_time.values if 'page_time' in df.columns else None
  read_completion_rates = df.read_completion_rate.values if 'read_completion_rate' in df.columns else None

  result = eval(df.duration.values, df[key].values, df[group_key].values, selected_uids, group=group, mode=mode,
                video_times=video_times, page_times=page_times,read_completion_rates=read_completion_rates,
                df=df)
        
  if online_result:
    # NOTICE for click and dur not dict_prefix here but later in evaluate_rank 
    pre = ''
    if mode == 'click':
      pre = 'Click'
    elif mode == 'dur':
      pre = 'Dur'
    info = 'info' if not FLAGS.model_name else FLAGS.model_name[:20]
    info = info if not pre else pre
    if pre:
      pre = f'{pre}/'
    importants = [info] + metrics
    result[info] = 'epoch:%.2f' % melt.epoch() if not FLAGS.valid_hour else FLAGS.valid_hour
    online_result[info] = online_result['info']
        
    online_result_ = online_result.copy()
    if pre:
      for key in online_result_:
        if pre + key in online_result_:
          online_result_[key] = online_result_[pre + key]

    diff = dict([(key, result[key] - online_result_[key]) for key in result if key != info and key in online_result_])
    diff[info] = 'diff'
    def print_(*args, **kwargs):
      logging.vlog(30, ' '.join("{}".format(a) for a in args), **kwargs)
 
    gezi.pprint_df(pd.DataFrame.from_dict([online_result_, result, diff]), 
                   [x for x in importants if x in result],
                   print_fn=print_, rename_fn=rename, format='%.4f')

    del result[info]

  return result

def evaluate_df(df, group=True, group_by_impression=False, key='pred', online_key='ori_lr_score',
                eval_all=True, eval_click=True, eval_duration=True,
                show_online=True, eval_online=False, online_result_only=False, train_valid=False):
  df = df.fillna(0.)
  if not group_by_impression:
    group_key = 'mid'
  else:
    group_key = 'impression'
    if group_key not in df.columns:
      df[group_key] = df.mid + '\t' + df.impression_time.astype(str)
    df = df[df.impression != 0]

  cb_users = set(FLAGS.cb_users.split(','))

  df_all = df
  df_noncold = df[~df.rea.astype(int).isin(cb_users)]
 
  df = df_all if eval_all else df_noncold
  if 'distribution' in df.columns:
    df_quality = df[df.distribution.isin(util.quality_set)]
  else:
    df_quality = pd.DataFrame()

  if 'num_tw_histories' not in df.columns:
    df['num_tw_histories'] = 0
    df['num_vd_histories'] = 0
  
  dfs = {}

  selected_uids = {}
  selected_uids['cold'] = set(df[df.rea.astype(int).isin(cb_users)].mid)
  if len(df_quality):
    selected_uids['quality'] = set(df_quality.mid)

  selected_uids['activity_-1'] = set(df[df.activity == -1].mid)
  selected_uids['activity_0'] = set(df[df.activity == 0].mid)
  selected_uids['activity_1'] = set(df[df.activity == 1].mid)
  selected_uids['activity_2'] = set(df[df.activity == 2].mid) 

  # max_histories = max(df.num_tw_histories.max(), df.num_vd_histories.max())
  max_histories = 100
  df['num_histories'] =  df.num_tw_histories + df.num_vd_histories  
  selected_uids['history_0'] = set(df[df.num_histories==0].mid)
  selected_uids['history_0_max'] = set(df[(df.num_histories>0)&(df.num_histories<max_histories)].mid)
  selected_uids['history_max'] = set(df[df.num_histories>=max_histories].mid)

  if FLAGS.eval_first_impression:
    df['impression'] = df.mid + '\t' + df.impression_time.astype(str)
    impressions = set(df[['mid', 'impression_time', 'impression']].groupby('mid', as_index=False).first().impression.values)
    df_top = df[df.impression.isin(impressions)]
    dfs['top'] = df_top

  if train_valid:
    global step 
    global is_first
    global online_result
    is_first_ = is_first
    
    step = melt.get_eval_step()

    if is_first or FLAGS.num_rounds > 1:
      is_first = False
      # global logger   
      logger = None
      if logger is None and FLAGS.write_metric_summary:
        logger = melt.get_summary_writer()

      base_name = 'base'
      if show_online:
        info = f'{base_name}({FLAGS.abtestids})'
        online_result = parse_base_file()
        # may be auc is np.nan or 0 or .. not valid online result
        if not online_result or 'auc' not in online_result or not(online_result['auc'] > FLAGS.min_online_auc):
          if eval_online:
            base_name = 'online'
            info = f'{base_name}({FLAGS.abtestids})'  
            try:
              online_result = evaluate_rank_(df[df[online_key] > 0], selected_uids, group=group, key=online_key, 
                                            group_by_impression=group_by_impression)
            except Exception:
              logging.warning(traceback.format_exc())
        
        # global baseline_logger
        baseline_logger = None 
        if baseline_logger is None and FLAGS.write_metric_summary and FLAGS.write_online_summary:
          baseline_log_dir = os.path.join(os.path.dirname(FLAGS.log_dir), base_name)
          baseline_logger = SummaryWriter(baseline_log_dir)
        
        if online_result and 'auc' in online_result and online_result['auc'] > FLAGS.min_online_auc:
          online_result['info'] = info
          # gezi.pprint_dict(online_result, importants, print_fn=logging.debug, rename_fn=rename)
          if online_result_only:
            del online_result['info']
            return online_result

    if online_result and baseline_logger is not None:
      for key_, val in online_result.items():
        if isinstance(val, float):
          baseline_logger.scalar(key_, val, step)
      
      for key_ in keys:
        if key_ in online_result:
          baseline_logger.scalar(f'all_/{key_}', online_result[key_], step)
    
    logging.debug('baseline_metrics:{}'.format(['%s:%.5f' % (name, val) for name, val in online_result.items() if not isinstance(val, str)]))

  result = evaluate_rank_(df, selected_uids, group=group, key=key, group_by_impression=group_by_impression)
  # score diff with online score
  result['inv_rate'] = gezi.metrics.inverse_ratio(df_all[online_key].values, df_all[key].values) 

  if train_valid:
    for key_ in keys:
      if key_ in result:
        logger.scalar(f'AAA_All/{key_}', result[key_], step)
        logger.scalar(f'AAA_all/{key_}', result[key_], step)
        if baseline_logger:
          if key_ in online_result:
            baseline_logger.scalar(f'AAA_All/{key_}', online_result[key_], step)

  if  eval_click:
    if 'pred_click' in df.columns:
      logging.debug('-------------------- evaluate click')
      result_click = evaluate_rank_(df, selected_uids, group=group, mode='click', key='pred_click',
                                    group_by_impression=group_by_impression)
    else:
      result_click = result.copy()
    if train_valid and logger is not None:
      for key_ in keys:
        if key_ in result_click:
          logger.scalar(f'AAA_Click/{key_}', result_click[key_], step)
          logger.scalar(f'AAA_click/{key_}', result_click[key_], step)
        if baseline_logger:
          if 'Click/' + key_ in online_result:
            baseline_logger.scalar(f'AAA_Click/{key_}', online_result['Click/' + key_], step)
    result_click = gezi.dict_prefix(result_click, 'Click/') 
    result.update(result_click)
  
  if eval_duration:
    if 'pred_dur' in df.columns:
      logging.debug('-------------------- evaluate dur')
      result_dur = evaluate_rank_(df, selected_uids, group=group, mode='dur', key='pred_dur',
                                  group_by_impression=group_by_impression)
    else:
      result_dur = result.copy()
    if train_valid and logger is not None:
      for key_ in keys:
        if key_ in result_dur:
          logger.scalar(f'AAA_Dur/{key_}', result_dur[key_], step)
          logger.scalar(f'AAA_dur/{key_}', result_dur[key_], step)
        if baseline_logger:
          if 'Dur/' + key_ in online_result:
            baseline_logger.scalar(f'AAA_Dur/{key_}', online_result['Dur/' + key_], step)
    result_dur = gezi.dict_prefix(result_dur, 'Dur/') 
    result.update(result_dur)

  if FLAGS.train_hour:
    result['other/train_hour'] = int(FLAGS.train_hour[-2:])
    result['other/train_day'] = int(FLAGS.train_hour[4:8])
  if FLAGS.valid_hour:
    result['other/valid_hour'] = int(FLAGS.valid_hour[-2:])
    result['other/valid_day'] = int(FLAGS.valid_hour[4:8])
  if FLAGS.l2_:
    result['loss/l2'] = FLAGS.l2_
  if FLAGS.params_:
    result['other/params'] = FLAGS.params_

  if FLAGS.total_time:
    result['perf/total_time'] = FLAGS.total_time
  if FLAGS.train_time:
    result['perf/train_time'] = FLAGS.train_time
  if FLAGS.valid_time:
    result['perf/valid_time'] = FLAGS.valid_time

  if train_valid and online_result:
    gezi.set('online_result', online_result)

  return result

def gen_inputs(x):
  keys = [
    'mid', 'docid', 'rea', 'product', 'distribution'
  ]
  for key in keys:
    x[key] = gezi.decode(x[key])

  uids, durations = x['mid'], x['duration']

  online_score_name = 'ori_lr_score'
  # turn back to use ori lr score as for video with no rerank 
  # if 'video' in FLAGS.train_input:
  #   online_score_name = 'lr_score'
  online_scores = x[online_score_name] 
  abtestids = x['abtestid']
  # Notice what you use should add on eval_keys otherwise will not find like rea
  reas = x['rea']
  reas = reas

  inputs = {
    'mid': uids,
    'duration': durations,
    'ori_lr_score': online_scores,
    'abtest': abtestids,
    'rea': reas,
    'page_time': ((x['impression_time'] - x['article_page_time']) / 60 / 60).astype(int),
    'product_data':  x['product'],
    'distribution': x['distribution'],
    'impression_time': x['impression_time'],
    'article_page_time': x['article_page_time'],
    'position': x['position'],
    'activity': x['user_active']
  }  
  
  if x['video_time'].max() > 0:
    inputs['video_time'] = x['video_time']
  if x['read_completion_rate'].max() > 0:
    inputs['read_completion_rate'] = x['read_completion_rate']
    
  return inputs

def evaluate_rank(y_true, y_pred, x, other={}, group=True, group_by_impression=False, product=None):
  inputs = gen_inputs(x)  
  inputs['y_ture'] = y_true
  inputs['pred'] = y_pred
 
  if other:
    if FLAGS.eval_click and 'prob_click' in other:
      inputs['pred_click'] = other['prob_click']
    if FLAGS.eval_dur and 'prob_dur' in other:
      inputs['pred_dur'] = other['prob_dur']

    inputs['num_tw_histories'] = other['num_tw_histories']
    inputs['num_vd_histories'] = other['num_vd_histories']

  df = pd.DataFrame(inputs)

  if 'product_data' in df.columns:
    df = df[df.product_data==FLAGS.eval_product]

  testids = set(map(int, FLAGS.abtestids.split(','))) if FLAGS.abtestids else set()
  if FLAGS.abtestids and (FLAGS.abtestids != 'all' and FLAGS.abtestids != 'none'):
    df = df[df.abtest.isin(testids)]

  return evaluate_df(df, group=group, 
                     group_by_impression=group_by_impression, 
                     eval_all=FLAGS.eval_all, 
                     eval_click=FLAGS.eval_click,
                     eval_duration=FLAGS.eval_dur,
                     show_online=FLAGS.show_online, 
                     eval_online=FLAGS.eval_online,
                     online_result_only=FLAGS.online_result_only,
                     train_valid=True)

class RankEvaluator(object): 
  def __init__(self): 
    self.first = True
  
  def eval_group(self):
    if FLAGS.eval_group is not None: 
      return FLAGS.eval_group

    if self.first:
      self.first = False
      return True 
    if melt.epoch() == FLAGS.num_epochs:
      return True 
      
    return False

  def __call__(self, y_true, y_pred, x, other):
    return evaluate_rank(y_true, y_pred, x, other, group=self.eval_group(), 
                         group_by_impression=FLAGS.eval_group_by_impression)

# NOTICE this is a must if you want get info from model other then from tfrecord X, so sess.run will deal these 
def out_hook(model): 
  if hasattr(model, 'prob_click') and hasattr(model, 'prob_dur'):
    m = dict(prob_click=model.prob_click, prob_dur=model.prob_dur)
  else:
    m = {}

  m['num_tw_histories'] = model.num_tw_histories
  m['num_vd_histories'] = model.num_vd_histories

  return m

# NOTICE labels is not used.. as we can get all info from durs but still for compat here
def valid_write(ids, labels, predicts, ofile, others):
  # this seqeunce is set by eval_keys in util.get_eval_fn_and_keys
  m = gen_inputs(ids)
  m.update({
     'docid': ids['docid'],
     'pred': predicts,
     'pred_click': others.get('prob_click', predicts),
     'pred_dur': others.get('prob_dur', predicts),
     'num_tw_histories': others.get('num_tw_histories', 0),
     'num_vd_histories': others.get('num_vd_histories', 0)
  })
  df = pd.DataFrame(m)
  # TODO maybe we need to store all infer result 
  testids = set(map(int, FLAGS.abtestids.split(','))) if FLAGS.abtestids else set()
  if testids:
    df = df[df.abtest.isin(testids)]
  
  df.to_csv(ofile, index=False)
