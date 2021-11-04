#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   nature_diff.py
#        \author   chenghuige  
#          \date   2019-12-28 08:12:06.956839
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from datetime import timedelta, datetime
import pandas as pd 

diff_spans = {
    8: [20191115, 20191121],
    15: [20191113, 20191118],
    16: [20191104, 20191110],
}     

def update_diff_spans(spans=None):
  global diff_spans
  if spans:
    for abid in spans:
      diff_spans[abid] = spans[abid]
      
  start_time = 1e23
  end_time = -1

  for key in diff_spans:
    val = diff_spans[key]
    s, e = val
    if s < start_time:
        start_time = s
    if e > end_time:
        end_time = e
    e = datetime.strptime(str(e), '%Y%m%d')
    e = (e + timedelta(1)).strftime('%Y%m%d')
    diff_spans[key][1] = int(e)
  return start_time, end_time

def calc_diff_hourly(df, abids, time_name='datetime'):
  base_all = df[df.abtest.isin([4,5,6])]

  base_times = sorted(list(set(base_all[time_name].values)))
  base = base_all.groupby(time_name).mean().transform(lambda x: x)
  base *= -1.
  base = base.assign(x=base_times)

  diff_res = {}
  for abid in abids:
    if not abid in diff_spans:
      print('missing diff span for', abid, file=sys.stderr)
      continue

    s, e = diff_spans[abid]
    s, e = diff_spans[abid]
    s = str(s * 100)
    e = str(e * 100)

    df_ = df[df[time_name] >= s]
    df_ = df_[df_[time_name] < e]
    df_ = df_[df_.abtest==abid]

    times = sorted(set(df_[time_name].values))

    df_ = df_.groupby(time_name).mean()
    base_ = base[base.x >= s]
    base_ = base_[base_.x < e]
    df_ = pd.concat([df_, base_], sort=False).groupby(time_name).sum().transform(lambda x: x)
    df_ = df_.assign(time=times)
    df_['key'] = list(map(lambda x: int(float(x) % 100), df_.time))
    df_ = df_.groupby('key').mean().transform(lambda x: x)
    df_ = df_.assign(hour=range(24))
    diff_res[abid] = df_
  return diff_res

def gen_diffs_hourly(dfs_all, names, abids):
  diffs = {}
  for name in names:
    diffs[name] = calc_diff_hourly(dfs_all[name], abids)
  return diffs
  
bases = {}
def calc_diff_daily(df, abids, time_name='date'):
  base_all = df[df.abtest.isin([4,5,6])]

  base_times = sorted(list(set(base_all[time_name].values)))
  base = base_all.groupby(time_name).mean().transform(lambda x: x)
  base = base.assign(x=base_times)
  bases[df.name] = base

  diff_res = {}
  ratio_res = {}
  for abid in abids:
    if not abid in diff_spans:
      print('missing diff span for', abid, file=sys.stderr)
      continue

    s, e = diff_spans[abid]
    s = str(s)
    e = str(e)

    df_ = df[df[time_name] >= s]
    df_ = df_[df_[time_name] < e]
    df_ = df_[df_.abtest==abid]
          
    df_mean = df_.mean()
    base_ = base[base.x >= s]
    base_ = base_[base_.x < e]
    base_mean = base_.mean()

    diff_mean = df_mean - base_mean 
    ratio_mean = diff_mean / base_mean
    diff_res[abid] = diff_mean
    ratio_res[abid] = ratio_mean
  return diff_res, ratio_res

def gen_diffs_daily(dfs_all, names, abids, stats=None):
  diffs = {}
  ratios = {}
  for name in names:
    diff, ratio = calc_diff_daily(dfs_all[name], abids)
    for abid in diff:
      diff[abid] = diff[abid].to_frame().T
      diff[abid].abtest = abid
      ratio[abid] = ratio[abid].to_frame().T
      ratio[abid].abtest = abid
    diff = pd.concat(diff)
    ratio = pd.concat(ratio)
    diff = diff.assign(name=name)
    ratio = ratio.assign(name=name)
    diffs[name] = diff
    ratios[name] = ratio
  if stats:
    diffs = pd.concat(diffs)[['name', 'abtest'] + stats] 
    ratios = pd.concat(ratios)[['name', 'abtest'] + stats] 
  return diffs, ratios

def gen_diffs(dfs_all, names, abids, mark):
  if mark == 'hourly':
    diffs = gen_diffs_hourly(dfs_all, names, abids)
  else:
    diffs, _ = gen_diffs_daily(dfs_all, names, abids)
  return diffs
