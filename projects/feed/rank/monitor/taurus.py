#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   taurus.py
#        \author   chenghuige  
#          \date   2019-12-28 08:02:55.863781
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import time
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine,Table,Column,Integer,String,MetaData,ForeignKey
from projects.feed.rank.monitor import natural_diff 
from projects.feed.rank.monitor.show import gen_figs
import gezi

def calc_stats(df):
    df['ctr'] = df['click'] / df['dis']
    try:
      df['real_ctr'] = df['click'] / df['real_dis']
    except Exception:
        pass
    df['read_ratio'] = df['click_user'] / df['dis_user']
    df['duration'] /= 60
    df['dur1'] = df['duration'] / df['dis_user'] 
    df['dur2'] = df['duration'] / df['click_user'] 
    df['finish_ratio'] = df['finished'] / df['click_back']
    df['read_files'] = df['click'] / df['dis_user']
    df['doc_dur'] = df['duration'] / df['click'] 
    df['refresh'] = df['refresh_times'] / df['dis_user']
    df['refresh2'] = df['refresh_times'] / df['click_user']

class Taurus(object):
  def __init__(self):
    self.engine = create_engine("mysql+pymysql://feed_monitor:FeedMonitor2018@feed.feed_monitor.rds.sogou:3306/feed_monitor", 
                                encoding="utf-8", echo=False)
    self.base_abids = [4, 5, 6] 

  def init_table(self, mark):
    self.mark = mark
    self.TABLE_NAME = f'feed_abtest_video_type_common_{mark}'

  def init_time(self, last_days=1, start_time=None, end_time=None):
    last_days = last_days + 0.2 if 'hourly' in self.TABLE_NAME else last_days + 1
    now = time.strftime('%Y%m%d%H', time.localtime(time.time()))
    self.END_TM = now
    before = (datetime.today() + timedelta(-last_days)).strftime('%Y%m%d%H')
    self.START_TM = before
    
    self.before, self.now = int(before), int(now)

    if self.start_time:
      self.before = self.start_time
    if self.end_time:
      self.now = self.end_time

    if start_time and start_time * 100 < int(self.START_TM):
      self.START_TM = str(start_time * 100)
    
    if end_time and end_time * 100 > int(self.END_TM):
      self.END_TM = str(end_time * 100)

    if 'daily' in self.TABLE_NAME:
        self.START_TM = self.START_TM[:8]
        self.END_TM = self.END_TM[:8] 

    self.time_name = 'datetime' if 'hourly' in self.TABLE_NAME else 'date'

  def init_sql(self, abids=None, product='sgsapp'):
    TABLE_NAME = self.TABLE_NAME
    START_TM, END_TM = self.START_TM, self.END_TM
    time_name = self.time_name

    if abids != None:
      self.abids = abids

    def get_abids_str(abids):
      abids = set(abids)
      abids.update(map(str, self.base_abids))
      abids_str = ','.join([f"'{x}'" for x in abids])
      abids_str = f"({abids_str})"
      return abids_str
    abids_str = get_abids_str(abids)

    if 'daily' in TABLE_NAME:
      START_TM = START_TM[:8]
      END_TM = END_TM[:8]
      
    sql = f'''SELECT * FROM {TABLE_NAME} 
                WHERE product = '{product}'
                AND abtest in {abids_str} 
                AND {time_name} BETWEEN '{START_TM}' AND '{END_TM}'
                AND video_type in ('0', '1', '2', 'total')
          '''

    self.sql = sql

  def init_cols(self):
    res = self.engine.execute(f"SHOW FULL COLUMNS FROM {self.TABLE_NAME}")
    self.columns = [x[0].replace('product', 'product_name') for x in res.fetchall()]

  def init(self, abids, last_days, mark='hourly', product='sgsapp', start_time=None, end_time=None, diff_spans=None):
    self.init_table(mark)
    self.init_cols()

    self.start_time = start_time
    self.end_time = end_time
    self.mark = mark
    self.abids = abids
    self.product = product
    self.diff_spans = diff_spans

    if diff_spans:
      start_time_diff, end_time_diff = natural_diff.update_diff_spans(diff_spans)

      if not start_time or start_time_diff < start_time:
        start_time = start_time_diff

      if not end_time or end_time_diff > end_time:
        end_time = end_time_diff

    self.init_time(last_days, start_time, end_time)

    self.keys = ['abtest', 'datetime', 'dis', 'real_dis', 'click', 
                 'dis_user', 'click_user', 'duration', 'click_back',
                 'ctr', 'real_ctr', 'read_ratio', 'dur1', 'dur2', 
                 'finish_ratio', 'read_files', 'doc_dur']
    self.names = ['quality', 'all', 'tuwen', 'video', 'small_video', 'video_related', 'rec'] \
                  if self.product == 'sgsapp' else ['all', 'tuwen', 'video', 'small_video', 'rec']
    self.stats = ['read_ratio', 'dur1', 'dur2', 'click', 'duration', 'refresh_times',  
                  'refresh', 'refresh2', 'click_user', 'read_files', 'doc_dur',
                  'dis_user', 'ctr', 'real_ctr','finish_ratio', 'praise', 'favor', 'share']

  def update(self):
    if not self.end_time:
      now = time.strftime('%Y%m%d%H', time.localtime(time.time()))

      self.END_TM = now
      self.now = int(now)

      self.init_sql(self.abids, self.product)

  def search(self, sql):
    print(sql, file=sys.stderr)
    res = self.engine.execute(sql)
    return res

  def run(self):
    timer = gezi.Timer('taurus run', True)
    #-------------------gen df
    self.update()
    
    time_name = self.time_name

    res_data = self.search(self.sql)

    df = pd.DataFrame.from_dict(res_data)
    df.columns = self.columns
    df = df[df.os=='total'][df.article_activity=='total'][df.video_activity=='total']
    self.df = df
    self.df1 = df
    if self.mark == 'daily':
      df = df.groupby([self.time_name, 'product_name', 'video_type', 'abtest', 'data_obj'], as_index=False).sum()
    
    def get_video_type(x):
      video_types = {
        '0': 'tuwen',
        '1': 'video',
        '2': 'small_video',
        'total': 'total'
      }
      return video_types.get(x, 'null')
    
    df['real_dis'] = np.maximum(df['real_dis'], 1)
    df['refresh_times'] = np.maximum(df['refresh_times'], 1)
    df['dis_user'] = np.maximum(df['dis_user'], 1)
    df['click_user'] = np.maximum(df['click_user'], 1)
    df['abtest'] = df['abtest'].astype(int)
    if 'date' in self.columns:
      df['date'] = df['date'].astype(int)
    if 'datetime' in self.columns:
      df['datetime'] = df['datetime'].astype(int)
    df['video_type'] = df['video_type'].apply(lambda x: get_video_type(x))
    df = df.fillna(0.)
    df = df.rename(columns={"video_type": "name"})
    df = df.sort_values(by=[time_name, 'name', 'data_obj', 'abtest'])

    calc_stats(df)
    tofloat = { 
                "click": float, "dis": float, "real_dis": float,
                "dis_user": float, "click_user": float, 
                "read_files": float, "duration": float, "refresh_times":float, 
                "praise": float, "favor": float, "share": float
              }
    df = df.astype(tofloat)
    cols1 = ['product_name', time_name, 'name', 'data_obj', 'abtest']
    cols2 = [x for x in df.columns if x not in cols1 and x not in ['video_activity', 'article_activity', 'os']]
    self.cols1, self.cols2 = cols1, cols2
    df = df[cols1 + cols2]
    df = df.fillna(0.)
    df.reset_index(drop=True, inplace=True)

    self.df_all = df

    #-------------------------------gen natural diff
    if self.diff_spans:
      self.natural_diff, self.natural_base, self.natural_ratio = natural_diff.gen_diffs(df, self.abids, self.mark, self.base_abids)
    else:
      self.natural_diff, self.natural_base, self.natural_ratio = None, None, None

    #--------------------------------deal with natural diff
    self.base = df[df.abtest.isin(self.base_abids)].groupby(['product_name', time_name, 'name', 'data_obj'], as_index=False).mean()
    self.base.reset_index(drop=True, inplace=True)
    self.df = df[~df.abtest.isin(self.base_abids)]
    self.df.reset_index(drop=True, inplace=True)
    diffs = {}
    ratios = {}
    # ratios2 = {}
    for abid in self.abids:
      df = self.df[self.df.abtest==abid]
      df.reset_index(drop=True, inplace=True)
      diff = pd.concat([df[cols1], df[cols2] - self.base[cols2]], 1)
      ratio = pd.concat([df[cols1], (df[cols2] - self.base[cols2]) / self.base[cols2]], 1)
      diffs[abid] = diff
      ratios[abid] = ratio
      # if diffs:
      #   nratio = self.natural_diff[self.natural_diff.abtest==abid]
      #   nratio.reset_index(drop=True, inplace=True)
      #   ratio2 = pd.concat([df[cols1], (df[cols2] - self.base[cols2]) / self.base[cols2] - nratio[cols2]], 1)
      #   ratios2[abid] = ratio2

    self.ratio = pd.concat(ratios)
    self.ratio.reset_index(drop=True, inplace=True)

    if diffs:
      self.diff = pd.concat(diffs)
      self.diff.reset_index(drop=True, inplace=True)

      # self.ratio2 = pd.concat(ratios2)
      # self.ratio2.reset_index(drop=True, inplace=True)

      # this is final improve or lose ratio with natural diff considered
      self.ratio2 = self.ratio.copy()
      if self.natural_ratio is not None:
        ratio = self.ratio
        ratio2 = self.ratio2
        nratio_ = self.natural_ratio
        base_ = self.base
        for index, row in self.df.iterrows():
          nratio = nratio_[nratio_.name==row['name']]
          nratio = nratio[nratio.data_obj==row['data_obj']]
          nratio = nratio[nratio.abtest==row['abtest']]
          if not len(nratio):
            continue
          base = base_[base_.name==row['name']]
          base = base[base.data_obj==row['data_obj']]
          base = base[base[time_name]==row[time_name]]
          for stat in self.stats:
            # try:
            self.ratio2.at[index, stat] = (row[stat] - base[stat].values[0]) / base[stat].values[0] - nratio[stat].values[0] 
            # except Exception:
            #   pass

    print(timer.elapsed(), file=sys.stderr)
    return self.df_all

  def show(self, name, quality=False, field=None, use_diff=True, relative_diff=True, return_figs=False, **kwargs):
    from plotly.graph_objs import Scatter,Layout
    import plotly
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    #setting offilne
    plotly.offline.init_notebook_mode(connected=True)

    if not use_diff:
      diffs, diff_ratios = None, None
    elif relative_diff:
      diffs, diff_ratios = None, self.natural_ratio
    else:
      diffs, diff_ratios = self.natural_diff, None
    
    figs = gen_figs(self.df_all, self.stats, self.abids, 
                    name=name, quality=quality, field=field, mark=self.mark, 
                    diffs=diffs, diff_ratios=diff_ratios, product=self.product, 
                    start_time=self.before, end_time=self.now, **kwargs)
    
    if return_figs:
      return figs

    for fig in figs:
      py.iplot(fig)

