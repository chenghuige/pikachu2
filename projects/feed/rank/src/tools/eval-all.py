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
import random
import pandas as pd
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('level', 100, '20 print info, 21 will not print info, 5 will print debug')
flags.DEFINE_integer('num_abids', 20, '')
flags.DEFINE_bool('cold_start', False, '')
flags.DEFINE_bool('offline', True, '')
flags.DEFINE_bool('online', True, '')
flags.DEFINE_string('ofile_offline', None, '')
flags.DEFINE_string('ofile_online', None, '')
flags.DEFINE_string('offline_abids', '', '')
flags.DEFINE_string('online_abids', '', '')
flags.DEFINE_integer('onlyid', None, '')
flags.DEFINE_string('hour', None, '')
flags.DEFINE_string('shour', None, '')
flags.DEFINE_string('ehour', None, '')
flags.DEFINE_integer('max_deal', 0, '')
flags.DEFINE_bool('group_by_impression', False, '')
flags.DEFINE_integer('num_processes', None, '')
flags.DEFINE_string('product', 'sgsapp', 'or newsme, shida')
flags.DEFINE_bool('force', False, '')

import gezi
logging = gezi.logging

import traceback
import pandas as pd
import pymp
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
import glob

from projects.feed.rank.src.evaluate import evaluate_df as eval

# TODO add inverse_ratio for online offline compare for offline first abid
def deal(score_file, online, header_online, offline, header_offline, parallel=False):
  # logging.info('score_file:', score_file)
  # assert online or offline
  if not online and not offline:
    return
  print('score_file:', score_file, file=sys.stderr)
  dir = os.path.dirname(score_file)
  mark = 'tuwen' if 'tuwen' in os.path.realpath(dir) else 'video'

  df = None
  try:
    df = pd.read_csv(score_file)
  except Exception:
    return

  if 'product_data' in df.columns:
    df = df[df.product_data==FLAGS.product]
  
  if not 'page_time' in df.columns:
    if not 'article_page_time' in df.columns:
      df['page_time'] = 0
    else:
      df['page_time'] = df['impression_time'] - df['article_page_time']
  
  if not 'abtestid' in df.columns:
    df = df.rename(columns={'abtest': 'abtestid'})
  
  offline_abid = None
  if FLAGS.offline_abids:
    offline_abid = int(FLAGS.offline_abids.split(',')[0])

  if len(df) == 0:
    return

  df_dict = {}

  base_ids = set([4,5,6])
  
  online_abids = set()
  if FLAGS.online_abids:
    online_abids = set([int(x) for x in FLAGS.online_abids.split(',')])

  if online:
    for i in range(FLAGS.num_abids):
      if not online_abids or i in online_abids:
        df_dict[i] = df[df.abtestid==i]
  
    # 456 is baseline online
    df_dict[456] = df[df.abtestid.isin(base_ids)]

  offline_abids = set()
  if offline:
    # 45600 is baseline offline
    abid_ = 456 * 100
    oabids = [abid_]
    if FLAGS.offline_abids:
      for id_ in FLAGS.offline_abids.split(','):
        abid_ = int(id_) * 100
        oabids.append(abid_)

    for abid_ in oabids:
      if abid_ != 456 * 100:
        df_dict[abid_] = df[df.abtestid==int(abid_ / 100)]
      else:
        df_dict[abid_] = df[df.abtestid.isin(base_ids)]

      offline_abids.add(abid_)

  abids = list(df_dict.keys())

  hour = os.path.basename(dir)
  dir = os.path.dirname(dir)

  if FLAGS.onlyid:
    abids = [FLAGS.onlyid]

  manager = Manager() 
  metrics = manager.dict()
  
  if not FLAGS.ofile_online:
    ofile = '%s/%s_metrics_online.csv' %(dir, FLAGS.product) 
  else:
    ofile = FLAGS.ofile_online
  ofile2 = '%s/%s/%s_metrics_online.csv' %(dir, hour, FLAGS.product) 
  if not FLAGS.ofile_offline:
    ofile_offline = '%s/%s_metrics_offline.csv' %(dir, FLAGS.product)
  else:
    ofile_offline = FLAGS.ofile_offline
  ofile_offline2 = '%s/%s/%s_metrics_offline.csv' %(dir, hour, FLAGS.product)
  
  if FLAGS.group_by_impression:
    ofile = ofile.replace('.csv', '_impression.csv')
    ofile2 = ofile2.replace('.csv', '_impression.csv')
    ofile_offline = ofile_offline.replace('.csv', '_impression.csv')
    ofile_offline2 = ofile_offline2.replace('.csv', '_impression.csv')
  
  def deal(i):
    abid = abids[i]
    df_dict[abid] = df_dict[abid][df_dict[abid].ori_lr_score >= 0]
    df_dict[abid].duration = df_dict[abid].duration.astype(int)
    
    key = 'ori_lr_score' if abid not in offline_abids else 'pred'

    results = eval(df_dict[abid], group=True, key=key, group_by_impression=FLAGS.group_by_impression,
                   eval_cold=FLAGS.eval_cold, eval_quality=FLAGS.eval_quality, 
                   eval_click=FLAGS.eval_click, eval_duration=FLAGS.eval_dur)
    results['mark'] = mark
    results['hour'] = hour
    results['abtest'] = abid
    metrics[abid] = results

  if parallel:
    with pymp.Parallel(len(abids)) as p:
      for i in tqdm(p.range(len(abids)), ascii=True):  
        deal(i)
  else:
    for i in tqdm(range(len(abids)), ascii=True):  
      deal(i)
    
  metric_values = [val for key, val in metrics.items() if key not in offline_abids]
  if metric_values:
    res = pd.DataFrame(metric_values)
    mode = 'a' if not header_online else 'w'
    # res.to_csv(ofile, index=False, mode=mode, header=header_online)
    res.to_csv(ofile2, index=False)

  if offline_abids:
    res = pd.DataFrame([metrics[x] for x in offline_abids])
    mode = 'a' if not header_offline else 'w'
    # res.to_csv(ofile_offline, index=False, mode=mode, header=header_offline)
    res.to_csv(ofile_offline2, index=False)

def main(_):
  # logging.init('/tmp', file='eval-all.html', level=FLAGS.level)
  gezi.check_cpu_only()
  
  input = sys.argv[1]

  if os.path.isfile(input):
    dir = os.path.dirname(input)

  dir = input

  mark = 'tuwen' if 'tuwen' in os.path.realpath(dir) else 'video'

  ofile = '%s/%s_metrics_online.csv' %(dir, FLAGS.product) 
  ofile2 = '%s/%s_metrics_offline.csv' %(dir, FLAGS.product)

  if FLAGS.group_by_impression:
    ofile = ofile.replace('.csv', '_impression.csv')
    ofile2 = ofile2.replace('.csv', '_impression.csv')

  # for online
  done_hours = set()
  # for offline
  done_hours2 = set()

  if os.path.exists(ofile):
    try:
      df = pd.read_csv(ofile)
      df.hour = df.hour.astype(str)
      done_hours.update(df.hour.values)
    except Exception:
      print('rm %s' % ofile, file=sys.stderr)
      os.system('rm -rf %s' % ofile)

  if os.path.exists(ofile2):
    try:
      df = pd.read_csv(ofile2)
      df.hour = df.hour.astype(str)
      done_hours2.update(df.hour.values)
    except Exception:
      print('rm %s' % ofile2, file=sys.stderr)
      os.system('rm -rf %s' % ofile2)

  if FLAGS.force:
    done_hours = set()
    done_hours2 = set()

  if os.path.isfile(input):
    score_files = [input]
  else:
    if not FLAGS.hour:
      score_files = ['%s/valid.csv' % x for x in glob.glob('%s/*' % dir) \
                      if os.path.isdir(x) and os.path.exists('%s/valid.csv' % x)]
    else:
      score_files = ['%s/%s/valid.csv' % (dir, FLAGS.hour)]
      if FLAGS.hour in done_hours:
        FLAGS.online = False
      if FLAGS.hour in done_hours2:
        FLAGS.offline = False

    def filter_(score_file):
      hour = os.path.basename(os.path.dirname(score_file))
      if FLAGS.shour and hour < FLAGS.shour:
        return False
      if FLAGS.ehour and hour > FLAGS.ehour:
        return False
      return True

    score_files = list(filter(filter_, score_files))

    score_files.sort(key=lambda x: -os.path.getmtime(x))

    if FLAGS.max_deal:
      score_files = score_files[:FLAGS.max_deal]

  if len(score_files) <= 4 or FLAGS.num_processes==1:
    is_first = True
    for i, score_file in tqdm(enumerate(score_files)):
      hour = os.path.basename(os.path.dirname(score_file))
      online = FLAGS.online and  hour not in done_hours
      offline = FLAGS.offline and hour not in done_hours2
      if not online and not offline:
        continue
      deal(score_file, 
           online=online,  
           header_online=(is_first and not done_hours),
           offline=offline,
           header_offline=(is_first and not done_hours2), 
           parallel=True)
      is_first = False
  else:
    hour = os.path.basename(os.path.dirname(score_files[0]))
    online = FLAGS.online and  hour not in done_hours
    offline = FLAGS.offline and hour not in done_hours2
    deal(score_files[0], 
         online=online,  
         header_online=(not done_hours),
         offline=offline,
         header_offline=(not done_hours2), 
         parallel=True)
    score_files = score_files[1:]
    ps = min(len(score_files), FLAGS.num_processes or cpu_count())
    with pymp.Parallel(ps) as p:
      for i in tqdm(p.range(len(score_files)), ascii=True):  
        hour = os.path.basename(os.path.dirname(score_files[i]))
        online = FLAGS.online and  hour not in done_hours
        offline = FLAGS.offline and hour not in done_hours2
        deal(score_files[i], 
             online=online,  
             header_online=(not done_hours),
             offline=offline,
             header_offline=(not done_hours2), 
             parallel=False)


if __name__ == '__main__':
  app.run(main)
  
