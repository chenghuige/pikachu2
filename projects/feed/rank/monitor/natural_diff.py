#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   nature_diff.py
#        \author   chenghuige  
#          \date   2019-12-28 08:12:06.956839
#   \Description  
# ==============================================================================
# Notice for groupby if you have date which is string and do mean it will auto drop date column
# for df1 - df2 then if you have date is string then will rasie error 
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from datetime import timedelta, datetime
import pandas as pd 
import gezi

diff_spans = {
    8: [20191115, 20191121],
    15: [20191113, 20191118],
    16: [20191104, 20191110],
    7: [20191230, 20200105]
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

def calc_diffs_hourly(df, abids, time_name='datetime', base_abids=[4, 5, 6]):
  base_all = df[df.abtest.isin(base_abids)]

  base_times = sorted(list(set(base_all[time_name].values)))
  base = base_all.groupby([time_name, 'name', 'data_obj'], as_index=False).mean()
  df_ori = df
  base_ori = base

  diffs, bases, ratios = {}, {}, {}
  for abid in abids:
    if not abid in diff_spans:
      print('missing diff span for', abid, file=sys.stderr)
      continue

    df = df_ori
    base = base_ori
    s, e = diff_spans[abid]
    s, e = diff_spans[abid]
    s = s * 100
    e = e * 100

    df = df[df[time_name] >= s]
    df = df[df[time_name] < e]
    df = df[df.abtest==abid]
    if not len(df):
      print('No data in df for diff span', s, e, file=sys.stderr)
      continue

    df = df.groupby([time_name, 'abtest', 'name', 'data_obj'], as_index=False).mean()
    base = base[base[time_name] >= s]
    base = base[base[time_name] < e]
    # need to reset index so as to substract between two dataframes
    df.reset_index(drop=True, inplace=True)
    base.reset_index(drop=True, inplace=True)
    cols1 = ['abtest', time_name, 'name', 'data_obj']
    cols2 = [x for x in df.columns if x not in cols1]
    df = df.fillna(0.)
    base.fillna(0.)
    # gezi.add_global('df', df)
    # gezi.add_global('base', base)
    # gezi.add_global('cols1', cols1)
    # gezi.add_global('cols2', cols2)
    df1 = pd.concat([df[cols1], df[cols2] - base[cols2]], 1)
    df2 = pd.concat([df[cols1],(df[cols2] - base[cols2]) / base[cols2]], 1)
    df1['hour'] = df1.datetime.apply(lambda x: int(float(x) % 100))
    df2['hour'] = df2.datetime.apply(lambda x: int(float(x) % 100))
    df = df1.groupby(['hour', 'name', 'data_obj'], as_index=False).mean()
    ratio = df2.groupby(['hour', 'name', 'data_obj'], as_index=False).mean()
    diffs[abid] = df
    bases[abid] = base
    ratios[abid] = ratio
  if diffs:
    diff = pd.concat(diffs)
    diff.reset_index(drop=True, inplace=True)
    base = pd.concat(bases)
    base.reset_index(drop=True, inplace=True)
    ratio = pd.concat(ratios)
    ratio.reset_index(drop=True, inplace=True)
  else:
    diff = None
    base = None
    ratio = None
  return diff, base, ratio
  
def calc_diffs_daily(df, abids, time_name='date', base_abids=[4, 5, 6]):
  base_all = df[df.abtest.isin(base_abids)]
  base = base_all.groupby([time_name, 'name', 'data_obj'], as_index=False).mean()
  df_ori = df
  base_ori = base

  diffs, bases, ratios = {}, {}, {}
  for abid in abids:
    if not abid in diff_spans:
      print('missing diff span for', abid, file=sys.stderr)
      continue

    s, e = diff_spans[abid]

    df = df_ori
    base = base_ori
    df = df[df[time_name] >= s]
    df = df[df[time_name] < e]
    df = df[df.abtest==abid]

    if not len(df):
      print('No data in df for diff span', s, e, file=sys.stderr)
      continue

    df_mean = df.groupby(['abtest', 'name', 'data_obj'], as_index=False).mean()
    base = base[base[time_name] >= s]
    base = base[base[time_name] < e]
    base_mean = base.groupby(['abtest', 'name', 'data_obj'], as_index=False).mean()
    base_mean['abtest'] = abid
    
    df_mean.reset_index(drop=True, inplace=True)
    base_mean.reset_index(drop=True, inplace=True)
    cols1 = ['abtest', 'name', 'data_obj']
    cols2 = [x for x in df_mean.columns if x not in cols1]
    diff_mean = pd.concat([df_mean[cols1], df_mean[cols2] - base_mean[cols2]], 1)
    ratio_mean = pd.concat([diff_mean[cols1], diff_mean[cols2] / base_mean[cols2]], 1)
    base_mean = base_mean[cols1 + cols2]
    diffs[abid] = diff_mean
    bases[abid] = base_mean
    ratios[abid] = ratio_mean
  if diffs:
    diff = pd.concat(diffs)
    ratio = pd.concat(ratios)
    base = pd.concat(bases)
    diff = diff.fillna(0.)
    ratio = ratio.fillna(0.)
    base = base.fillna(0.)
    diff = diff[cols1 + cols2]
    ratio = ratio[cols1 + cols2]
    base = base[cols1 + cols2]
    diff.reset_index(drop=True, inplace=True)
    ratio.reset_index(drop=True, inplace=True)
    base.reset_index(drop=True, inplace=True)
  else:
    print('No diff data', file=sys.stderr)
    diff = None
    base = None
    ratio = None
  
  return diff, base, ratio

def gen_diffs(dfs_all, abids, mark, base_abids=[4, 5,6]):
  if mark == 'hourly':
    return calc_diffs_hourly(dfs_all, abids, base_abids=base_abids)
  else:
    return calc_diffs_daily(dfs_all, abids, base_abids=base_abids)
