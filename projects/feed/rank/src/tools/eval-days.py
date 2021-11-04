#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval-days.py
#        \author   chenghuige  
#          \date   2020-01-25 08:22:17.617530
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np 
import random
import traceback
import pandas as pd
import collections
from datetime import datetime
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('level', 100, '20 print info, 21 will not print info, 5 will print debug')
flags.DEFINE_string('start_day', None, '')
flags.DEFINE_string('end_day', None, '')
flags.DEFINE_integer('days_', None, '')
flags.DEFINE_string('day', None, '')
flags.DEFINE_integer('first_n', None, '')
flags.DEFINE_integer('last_n', None, '')
flags.DEFINE_string('models', None, '')
flags.DEFINE_string('type', 'offline', '')
flags.DEFINE_integer('num_processes', 12, '')
flags.DEFINE_string('product', 'sgsapp', 'or newsme, shida')
flags.DEFINE_string('parallel', 'model', '')
flags.DEFINE_bool('force_update', False, '')
flags.DEFINE_integer('min_hours', 20, '')
flags.DEFINE_integer('eval_step', None, '')
flags.DEFINE_bool('parallel_read', None, '')
flags.DEFINE_bool('tfrecord_base', False, '')
flags.DEFINE_bool('tfrecord_base_only', False, '')
flags.DEFINE_bool('eval_click_', True, '')
flags.DEFINE_bool('eval_dur_', True, '')
flags.DEFINE_bool('group_by_impression', True, '')
# flags.DEFINE_string('abtestids', '4,5,6', '')

#  CUDA_VISIBLE_DEVICES=-1 python ../tools/eval-days.py /search/odin/publicData/CloudS/libowei/rank_online/infos/video --day=20200516 --models=8 --force_update --type=online 
# nc eval-days.py . --day 20200516 --models=dlrm-din

import gezi
logging = gezi.logging

import traceback
import pandas as pd

use_pymp = True
try:
  import pymp
except Exception:
  use_pymp = False
  
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
import glob

from gezi import SummaryWriter
from projects.feed.rank.src.evaluate import evaluate_df as eval
from projects.feed.rank.src.evaluate import keys

root = None

def gen_df(model, day, parallel=False):
  if FLAGS.parallel_read == False:
    parallel = False
  if model:
    if FLAGS.type == 'offline':
      pattern = f'{root}/{model}/infos/{day}*/valid.csv'
    else:
      pattern = f'{root}/{model}/{day}*/valid.csv'
  else:
    pattern = f'{root}/infos/{day}*/valid.csv'
  files = glob.glob(pattern)

  if not files:
    return None
  testids = set(map(int, FLAGS.abtestids.split(','))) if FLAGS.abtestids else set()
  # print('testids', testids)
  def _deal(df):
    # print('len(df)', len(df))
    if not 'abtestid' in df.columns:
      df = df.rename(columns={'abtest': 'abtestid'}) 
    if testids:
      df = df[df.abtestid.isin(testids)]
    # print('len(df) of abtestids', testids, len(df))
    if 'product_data' in df.columns:
      df = df[df.product_data==FLAGS.product]
    # print('len(df) of product', FLAGS.product, len(df))
    return df
  if not parallel:
    dfs = []
    for file in tqdm(files, desc=model, ascii=True):
      hour = os.path.basename(os.path.dirname(file))
      df = []
      # try:
      df = pd.read_csv(file)
      df = _deal(df)
      # except Exception:
      #   pass
      if len(df):
        df['hour'] = hour
        dfs += [df]
  else:
    dfs = Manager().list()
    ps = min(len(files), cpu_count())
    with pymp.Parallel(ps) as p:
      for i in tqdm(p.range(len(files)), total=len(files), desc=model, ascii=True):
        file = files[i]
        hour = os.path.basename(os.path.dirname(file))
        # try:
        df = pd.read_csv(file)
        df = _deal(df)
        # except Exception:
        #   pass
        if len(df):
          df['hour'] = hour
          dfs.append(df)
    dfs = list(dfs)
  
  if not dfs:
    return None
  df = pd.concat(dfs)
  df['model'] = model
  df['hours'] = len(dfs)
  df['date'] = day
  df['online_score'] = df.position.apply(lambda x: 20 - x)
  return df

def deal(df):
  print('len(df)', len(df))
  if df is not None and len(df) >= FLAGS.min_hours:
    day = df.date.values[0]
    model = df.model.values[0]
    if model:
      ofile = f'{root}/{model}/infos_day/{day}/metrics.csv'
    else:
      ofile = f'{root}/infos_day/{day}/metrics.csv'
    if FLAGS.group_by_impression:
      ofile = ofile.replace('metrics.csv', 'metrics_impression.csv')
    odir = os.path.dirname(ofile)
    os.system(f'mkdir -p {odir}')
    if gezi.non_empty(ofile):
      try:
        df2 = pd.read_csv(ofile)
        if df2.hours.values[0] == df.hours.values[0] and not FLAGS.force_update:
          print('model:', model, 'day:', day, 'hours:', df.hours.values[0], 'exists and do nothing', file=sys.stderr)
          return 
      except Exception:
        pass
    gezi.system(f'mkdir -p {os.path.dirname(ofile)}')
    print('Deal model:', model, 'day:', day, 'hours:', df.hours.values[0], file=sys.stderr)
    kwargs = dict(group_by_impression=FLAGS.group_by_impression, eval_click=FLAGS.eval_click_, eval_duration=FLAGS.eval_dur_)
    if not 'page_time' in df.columns:
      if 'article_page_time' not in df.columns:
        df['page_time'] = 0
      else:
        df['page_time'] = ((df['impression_time'] - df['article_page_time']) / 60 / 60).astype(int)
    if not 'duration' in df.columns:
      df = df.rename(columns={'dur': 'duration'})
    if not 'activity' in df.columns:
      df = df.rename(columns={'user_active': 'activity'})
    try:
      evr = eval(df, **kwargs)
      evr['hours'] = df.hours.values[0]
      evr['model'] = df.model.values[0]
      evr['date'] = day
      df = pd.DataFrame([evr])
      df.to_csv(ofile, index=False)
      return df
    except Exception:
      print(traceback.format_exc(), file=sys.stderr)
      print(f'Error in generate {ofile}', file=sys.stderr)
    return None

def gen_dfs(models, day, parallel=False):
  dfs = []
  for model in models:
    df = gen_df(model, day, parallel)
    if df is not None:
      dfs += [df]
  if not dfs:
    return dfs
  dfs = sorted(dfs, key=lambda x: -x.hours.values[0])
  if FLAGS.tfrecord_base:
    base = dfs[0].copy()
    base['pred'] = base['ori_lr_score']
    base['model'] = 'online' 
    dfs += [base]
  if FLAGS.tfrecord_base_only:
    dfs = dfs[:1]
  return dfs
 
def main(_):
  gezi.check_cpu_only()
  
  input_dir = sys.argv[1]
  
  global root
  root = input_dir
  
  if FLAGS.tfrecord_base_only:
    FLAGS.tfrecord_base = True

  if not use_pymp:
    FLAGS.parallel_read = False
    if FLAGS.parallel == 'day':
      FLAGS.parallel = 'none'
    
  if FLAGS.type == 'offline':
    # assume model is like v15/fm v15/base
    # info dir is like v15/fm/infos/
    model_dir_pattern = f'{input_dir}/*'
    dir_pattern = f'{input_dir}/*/infos/*' 
  elif FLAGS.type == 'online':
    FLAGS.parallel_read = False
    FLAGS.parallel = 'none'
    # assume input is like /search/odin/publicData/CloudS/rank/infos/tuwen 
    # model is like /search/odin/publicData/CloudS/rank/infos/tuwen/15
    model_dir_pattern = f'{input_dir}/*'
    dir_pattern = f'{input_dir}/*/*'
    
  if FLAGS.models:
    models = FLAGS.models.split(',')
  else:
    models = [os.path.basename(x) for x in glob.glob(model_dir_pattern)]
    
  print('models:', models, file=sys.stderr)
  
  def dir_ok(dir):
    if not models:
      return True
    for model in models:
      if f'/{model}/' in dir:
        return True
    return False    
  
  dirs = [x for x in glob.glob(dir_pattern) if dir_ok(x)]
  hours = [os.path.basename(x) for x in dirs]
  days = [x[:-2] for x in hours]
  
  if FLAGS.eval_step:
    assert len(models) == 1
    model = models[0]
    dirs2 = [x for x in dirs if os.path.exists(f'{x}/valid.csv')]
    hours2 = [os.path.basename(x) for x in dirs2]
    days2 = [x[:-2] for x in hours2]
    m = collections.defaultdict(int)
    for day in days:
      m[day] += 1
    m2 = collections.defaultdict(int)
    for day in days2:
      m2[day] += 1

    hour = hours[-1][-2:]

    days = sorted(list(set(days)))

    writer = SummaryWriter(f'{input_dir}/{model}', is_tf=False)
    for i, day in enumerate(days):
      # print(f'In eval-days.py try deal day:{day} {i} {m[day]} {m2[day]} num_days now:', len(days), file=sys.stderr)
      if i == len(days) - 1 and m2[day] != 24:
        print(f'Exit eval-days.py day:{day} {i} {m[day]} {m2[day]} hour:{hour} i:{i} num_days now:{len(days)}', file=sys.stderr)
        break
      os.system(f'mkdir -p {input_dir}/{model}/infos_day/{day}')
      lock_file = f'{input_dir}/{model}/infos_day/{day}/metrics.lock'
      if (m2[day] == 24 or i < len(days) - 2 or (m[day] == m2[day] and hour >= '01') or hour >= '06') \
          and m2[day] >= FLAGS.min_hours \
          and not os.path.exists(f'{input_dir}/{model}/infos_day/{day}/metrics.csv') \
          and not os.path.exists(lock_file):
        os.system(f'touch {lock_file}')
        print(f'Deal eval-days.py day:{day} {i} {m[day]} {m2[day]} hour:{hour} i:{i} num_days now:{len(days)}', file=sys.stderr)
        df = None
        try:
          dfs = gen_dfs(models, day, parallel=True)
          df = deal(dfs[0].copy())
        except Exception:
          pass
        os.system(f'rm -rf {lock_file}')
        if df is not None and len(df):
          walltime=datetime.strptime(day + '00', '%Y%m%d%H').timestamp()
          for key in df.columns:
            val = df[key].values[0]
            if not isinstance(val, str):
              if key in keys:
                prefix = 'AAA_Day/' 
                if key.istitle():
                  prefix = 'AAA_Day_'
                writer.scalar(f'{prefix}{key}', val, i + 1, walltime=walltime)
              writer.scalar(f'Day_{key}', val, i + 1, walltime=walltime)
          if FLAGS.tfrecord_base:
            df = dfs[0]
            df['pred'] = df['ori_lr_score']
            df['model'] = 'online'
            deal(df)
        else:
          print(f'eval day {day} fail return df none')
    exit(0)

  days = sorted(list(set(days)))

  if FLAGS.day:
    days = [FLAGS.day]
  else:
    start = FLAGS.start_day
    end = FLAGS.end_day
    if FLAGS.days_ and not(start and end):
      if start:
        end = gezi.DateTime(start).add(FLAGS.days_)
      else:
        start = gezi.DateTime(end).add(-FLAGS.days_)
    def day_ok(day, start, end):
      if start and day < start:
        return False
      if end and day > end:
        return False
      return True
    days = [day for day in days if day_ok(day, start, end)]
  
  assert not (FLAGS.first_n and FLAGS.last_n)
  if FLAGS.first_n:
    days = days[:FLAGS.first_n]
  elif FLAGS.last_n:
    days = days[-FLAGS.last_n:]
  if FLAGS.parallel == 'none':
    for day in tqdm(days, desc='day', ascii=True):
      dfs = gen_dfs(models, day, parallel=True)
      if dfs:
        print('day:', day, 'dfs:', len(dfs), [x.model.values[0] for x in dfs], file=sys.stderr)
        for df in dfs:
          deal(df)
  elif FLAGS.parallel == 'model':
    for day in tqdm(days, desc='day', ascii=True):
      dfs = gen_dfs(models, day, parallel=True)
      if dfs:
        print('day:', day, 'dfs:', len(dfs), [x.model.values[0] for x in dfs], file=sys.stderr)
        ps = min(len(dfs), cpu_count())
        with Pool(ps) as p:
          p.map(deal, dfs)
  elif FLAGS.parallel == 'day':
    ps = min(len(days), cpu_count())
    with pymp.Parallel(ps) as p:
      for i in tqdm(p.range(len(days)), desc='day', ascii=True):
        dfs = gen_dfs(models, days[i])
        for df in dfs:
          deal(df) 
  else:
    # Has problem 
    dfs = Manager().list()
    ps = min(len(days), cpu_count())
    with pymp.Parallel(ps) as p:
      for i in tqdm(p.range(len(days)), desc='day', ascii=True):
        dfs += gen_dfs(models, days[i])
    ps = min(len(dfs), cpu_count())
    with pymp.Parallel(ps) as p:
      for i in tqdm(p.range(len(dfs)), desc='dfs', ascii=True):
        deal(dfs[i])
    
if __name__ == '__main__':
  app.run(main)
  
