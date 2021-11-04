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
import pandas as pd
from sqlalchemy import create_engine,Table,Column,Integer,String,MetaData,ForeignKey
from projects.common.monitor import natural_diff 
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

class Taurus(object):
  def __init__(self):
    self.engine = create_engine("mysql+pymysql://feed_monitor:FeedMonitor2018@feed.feed_monitor.rds.sogou:3306/feed_monitor", 
                                encoding="utf-8", echo=False) 

  def init_table(self, mark):
    self.mark = mark
    if mark == 'hourly':
      self.TABLE_NAME = "feed_abtest_hourly"
      self.TABLE_NAME_REL = "feed_relative_hourly"
    else:
      self.TABLE_NAME = "feed_abtest_daily"
      self.TABLE_NAME_REL = "feed_relative"  

  def init_time(self, last_days=1, start_time=None, end_time=None):
    last_days = last_days + 0.2 if 'hourly' in self.TABLE_NAME else last_days + 1
    now = time.strftime('%Y%m%d%H', time.localtime(time.time()))
    self.END_TM = now
    before = (datetime.today() + timedelta(-last_days)).strftime('%Y%m%d%H')
    self.START_TM = before
    
    self.before, self.now = before, now

    if self.start_time:
      self.before = str(self.start_time)
    if self.end_time:
      self.now = str(self.end_time)

    if start_time and start_time * 100 < int(self.START_TM):
      self.START_TM = str(start_time * 100)
    
    if end_time and end_time * 100 > int(self.END_TM):
      self.END_TM = str(end_time * 100)

    if 'daily' in self.TABLE_NAME:
        self.START_TM = self.START_TM[:8]
        self.END_TM = self.END_TM[:8] 

    self.time_name = 'datetime' if 'hourly' in self.TABLE_NAME else 'date'

  def init_sql(self, abids=None, product='sgsapp'):
    data_obj = ['recommend_ge6511_weight_mean_article',
                'recommend_ge6511_weight_mean_video', 'recommend_ge6511_weight_mean_small_video',
                'recommend_ge6511_weight_mean','ge6511_weight_mean','quality']
    data_obj_str = "('" + ("','").join(data_obj) + "')"

    TABLE_NAME, TABLE_NAME_REL = self.TABLE_NAME, self.TABLE_NAME_REL

    if abids != None:
      self.abids = abids

    abIds = set(map(str, self.abids))
    abIds.update(['4', '5', '6'])

    abIds_str = "(" + (",").join(abIds) + ")"
    abIds_str = abIds_str.replace("'","") 

    # '201910290100' AND '201910290900'
    start_tm_str, end_tm_str = f"'{self.START_TM}'", f"'{self.END_TM}'"  
    sql = f"SELECT * FROM {TABLE_NAME} "
    sql += f"WHERE product = '{product}' AND data_obj in {data_obj_str} AND " if self.product == 'sgsapp' else f"WHERE product = '{product}' AND "
    sql += f"abtest in {abIds_str} AND "
    sql += f"{self.time_name} BETWEEN  {start_tm_str}  AND  {end_tm_str}"
    if 'daily' in TABLE_NAME:
        sql += " AND user_obj = 'total' AND os = 'total'"
    print ("sql_search:===========",sql, file=sys.stderr)

    sql_rel = sql.replace(TABLE_NAME, TABLE_NAME_REL). \
              replace(f"AND data_obj in {data_obj_str}", '').replace("AND os = 'total'","") if self.product == 'sgsapp' else None

    print("sql_rel:==============", sql_rel, file=sys.stderr)

    self.sql = sql
    self.sql_rel = sql_rel

  def init_cols(self):
    res = self.engine.execute(f"SHOW FULL COLUMNS FROM {self.TABLE_NAME}")
    self.columns = [x[0] for x in res.fetchall()]
    res_rel = self.engine.execute(f"SHOW FULL COLUMNS FROM {self.TABLE_NAME_REL}")
    self.columns_rel = [x[0] for x in res_rel.fetchall()]

  def init(self, abids, last_days, mark='hourly', product='sgsapp', start_time=None, end_time=None, use_natural_diff=True, diff_spans=None):
    self.init_table(mark)
    self.init_cols()

    self.start_time = start_time
    self.end_time = end_time
    self.mark = mark
    self.abids = abids
    self.product = product

    if use_natural_diff or diff_spans:
      start_time_diff, end_time_diff = natural_diff.update_diff_spans(diff_spans)
      self.use_natural_diff = use_natural_diff

      if not start_time or start_time_diff < start_time:
        start_time = start_time_diff

      if not end_time or end_time_diff > end_time:
        end_time = end_time_diff

    self.init_time(last_days, start_time, end_time)
    self.init_sql(abids, product)

    self.keys = ['abtest', 'datetime', 'dis', 'real_dis', 'click', 
                 'dis_user', 'click_user', 'duration', 'click_back',
                 'ctr', 'real_ctr', 'read_ratio', 'dur1', 'dur2', 
                 'finish_ratio', 'read_files', 'doc_dur']
    self.names = ['quality', 'all', 'tuwen', 'video', 'small_video', 'video_related', 'rec'] \
                  if self.product == 'sgsapp' else ['all', 'tuwen', 'video', 'small_video', 'rec']
    self.stats = ['read_ratio', 'dur1', 'dur2', 'click', 'duration', 'refresh_times',  
                  'refresh', 'click_user', 'read_files', 'doc_dur',
                  'dis_user', 'ctr', 'real_ctr','finish_ratio', 'praise', 'favor', 'share']

  def update(self):
    if not self.end_time:
      now = time.strftime('%Y%m%d%H', time.localtime(time.time()))

      self.END_TM = now
      self.now = now

      self.init_sql(self.abids, self.product)

  def search(self, sql):
    res = self.engine.execute(sql)
    res_data = res.fetchall()
    return res_data

  def run(self):
    timer = gezi.Timer('taurus run', True)
    self.update()

    res_data = self.search(self.sql)
    if self.product == 'sgsapp':
      res_data_rel = self.search(self.sql_rel)

    df = pd.DataFrame.from_dict(res_data)
   
    df_rel = pd.DataFrame.from_dict(res_data_rel)  if self.product == 'sgsapp' else None
    
    df.columns = self.columns
    df = df.sort_values(by=[self.time_name])
    if df_rel is not None:
      df_rel.columns = self.columns_rel
      df_rel = df_rel.sort_values(by=[self.time_name])
      df_rel['real_dis'] = [1] * len(df_rel)
      df_rel['refresh_times'] = [1] * len(df_rel)
      df_rel['abtest'] = df_rel['abtest'].astype(int)

    calc_stats(df)
    if df_rel is not None:
      calc_stats(df_rel)

    tuwen = df[df.data_obj=='recommend_ge6511_weight_mean_article'] if self.product == 'sgsapp' else df[df.data_obj=='recommend_article']
    tuwen.name = 'tuwen'
    video = df[df.data_obj=='recommend_ge6511_weight_mean_video'] if self.product == 'sgsapp' else df[df.data_obj=='recommend_video']
    video.name = 'video'
    small_video = df[df.data_obj=='recommend_ge6511_weight_mean_small_video'] if self.product == 'sgsapp' else df[df.data_obj=='recommend_small_video']
    small_video.name = 'small_video'
    rec = df[df.data_obj=='recommend_ge6511_weight_mean'] if self.product == 'sgsapp' else df[df.data_obj=='recommend']
    rec.name = 'rec'
    all = df[df.data_obj=='ge6511_weight_mean'] if self.product == 'sgsapp' else df[df.data_obj=='total']
    all.name = 'all'
    quality = df[df.data_obj=='quality'] if self.product == 'sgsapp' else df[df.data_obj=='total']
    quality.name = 'quality'

    if df_rel is not None:
      tuwen_rel = df_rel[df_rel.video_type=='0']
      tuwen_rel.name = 'tuwen_related'
      video_rel = df_rel[df_rel.video_type=='1']
      video_rel.name = 'video_rel'
      all_rel = df_rel[df_rel.video_type=='all']
      all_rel.name = 'all_related'

    dfs_all = dict(
                  tuwen=tuwen,
                  video=video,
                  small_video=small_video,
                  rec=rec,
                  all=all,
                  quality=quality)

    if df_rel is not None:
      dfs_all.update(dict(
                  tuwen_related=tuwen_rel,
                  video_related=video_rel,
                  all_related=all_rel))

    if self.use_natural_diff:
      dfs = {}
      for name in self.names:
          df = dfs_all[name]
          df = df[df[self.time_name] >= self.before]
          df = df[df[self.time_name] <= self.now]
          dfs[name] = df
          dfs[name].name = name
      self.dfs = dfs

      self.diffs = natural_diff.gen_diffs(dfs_all, self.names, self.abids, self.mark)
    
      print(timer.elapsed(), file=sys.stderr)
      return self.dfs, self.diffs
    else:
      self.dfs = dfs_all
      print(timer.elapsed(), file=sys.stderr)
      return self.dfs, None

