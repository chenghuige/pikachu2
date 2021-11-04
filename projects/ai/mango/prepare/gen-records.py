#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   csv2records.py
#        \author   chenghuige  
#          \date   2020-04-12 17:56:50.100557
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm 
from sklearn.utils import shuffle
import pyarrow.parquet as pq
import time
from datetime import datetime
import json
import pickle
import ast
import traceback
import glob
from collections import defaultdict

import gezi, melt
import tensorflow as tf
from gezi import hash_int64 as hash
logging = gezi.logging

from projects.ai.mango.src.config import *

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_string('mark', 'train', 'train or eval')
flags.DEFINE_integer('num_records', 32, '')
#flags.DEFINE_integer('day', 30, '')
# flags.DEFINE_bool('toy', False, '')
flags.DEFINE_integer('seed_', 12345, '')
flags.DEFINE_bool('force', True, '')
flags.DEFINE_bool('lm', False, '')

flags.DEFINE_bool('gen_stars_corpus', False, '')

df = None
odir = None
vocabs = {}
vinfos = {}
timestamp = None
udf = None

d = Manager().dict()

def get_day(timestamp):
  x = time.localtime(timestamp)
  return x.tm_mday

def get_year_mon_day(timestamp):
  x = time.localtime(timestamp)
  return x.tm_year, x.tm_mon, x.tm_mday

# 时间穿越或者当天的历史去掉 测试集合没有当天历史
# 只留T-1以及之前的和测试集合看齐 
def is_badtime(x, timestamp):
  if get_year_mon_day(x) >= get_year_mon_day(timestamp):
    return True

def gen_context_feats(row):
  feats = {}
  cols = context_cols
  for i in range(len(cols) - 1):
    for j in range(i+1, len(cols)):
      feats[f'{cols[i]}_{cols[j]}'] = hash(f'{row[cols[i]]}_{row[cols[j]]}')

  return feats

def gen_item_feats(row):
  feats = {}
  cols = item_cols
  for i in range(len(cols) - 1):
    for j in range(i+1, len(cols)):
      feats[f'{cols[i]}_{cols[j]}'] = hash(f'{row[cols[i]]}_{row[cols[j]]}')
  return feats

# cross feats change to cross_
def gen_cross_feats(row):
  feats = {}
  for context_col in context_cols:
    for item_col in item_cols:
      if item_col in ignored_cols and context_col in ignored_cols:
        continue
      feats[f'cross_{context_col}_{item_col}'] = hash(f'{row[context_col]}_{row[item_col]}')

  # l = []
  # for context_col in context_cols:
  #   for star in row['stars']:
  #     l += hash(f'{row[context_col]_{star}}')
  # feats['match_stars'] = l
  return feats

def merge_uv(index):
  total = len(udf)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  udf_ = udf.iloc[start:end]

  watch_vids_list = []
  watch_times_list = []
  cids_list = []
  class_ids_list = []
  second_classes_list = []
  is_intacts_list = []
  titles_list = []
  all_stars_list = []
  all_stars_str_list = []
  stars_list = []
  first_stars_list = []
  hits_list = []
  durs_list = []
  freshes_list = []

  last_stars_list = []
  last_title_list = []
  
  context_year_mon_day =  get_year_mon_day(timestamp)
  for _, row in tqdm(udf_.iterrows(), total=len(udf_), ascii=True):
    watches = row['watch']
    # watches2 = sorted(watches, reverse=True)
    watch_vids_ = []
    watch_times_ = []
    cids = []
    class_ids = []
    second_classes = []
    is_intacts = []
    
    titles = []
    m_title = defaultdict(float)
    stars = []
    m_star = defaultdict(int)
    all_stars = set()
    first_stars = []
    durs = []
    freshes = []

    last_stars = [0]
    last_title = [0]
    
    hits = 0
    i = 0
    ok = False
    for time_, vid in watches:
      if not ok and get_year_mon_day(time_) >= context_year_mon_day:
        # print(get_year_mon_day(time_), context_year_mon_day)
        continue
      else:
        ok = True
      i += 1
      watch_vids_ += [vid]
      watch_times_ += [time_]
      vrow = vinfos[vid] if vid in vinfos else None
      cids += [vrow['cid']] if vrow is not None else [0]
      class_ids += [vrow['class_id']] if vrow is not None else [0]
      second_classes += [vrow['second_class']] if vrow is not None else [0]
      is_intacts += [vrow['is_intact']] if vrow is not None else [0]
      
      if vrow is not None:
        if i == 1:
          last_stars = [vocabs['stars'].id(x) for x in vrow['stars']]
          try:
            last_title = [vocabs['words'].id(x) for x in vrow['title']]
          except Exception:
            last_title = [1]

        durs += [vrow['duration']]
        freshes += [max(time_ - vrow['timestamp'], 1)]
        
        hits += 1
        title = vrow['title']
        if title != 0:
          try:
            words = title.split(',')
            for word in words:
              m_title[word] += 1. / len(words)
          except Exception:
            pass
        stars = vrow['stars2']
        if stars != 0:
          try: # TODO why still has float ?..
            stars = stars.split(',')
            for star in stars[:2]:
              m_star[star] += 1
            first_stars += [stars[0]]
            for star in stars:
              all_stars.add(star)
          except Exception:
            first_stars += ['UNK']
        else:
          first_stars += ['UNK']
      else:
        durs += [0.]
        freshes += [0]

    watch_vids_list += [[vocabs['vid'].id(x) for x in watch_vids_]]
    watch_times_list += [watch_times_]
    cids_list += [[vocabs['cid'].id(x) for x in cids]]
    class_ids_list += [[vocabs['class_id'].id(x) for x in class_ids]]
    second_classes_list += [[vocabs['second_class'].id(x) for x in second_classes]]
    is_intacts_list += [[vocabs['is_intact'].id(x) for x in is_intacts]]

    titles = [x[0] for x in sorted(m_title.items(), key=lambda kv: -kv[1])]
    titles_list += [[vocabs['words'].id(x) for x in titles[:100]]]
    stars = [x[0] for x in sorted(m_star.items(), key=lambda kv: -kv[1])]

    stars_list += [[vocabs['stars'].id(x) for x in stars[:100]]]
    all_stars_list += [[vocabs['stars'].id(x) for x in all_stars]]
    all_stars_str_list += [list(all_stars)]
    first_stars_list += [[vocabs['stars'].id(x) for x in first_stars]]

    durs_list += [[float(x) for x in durs]]
    freshes_list += [[int(x) for x in freshes]]
    hits_list +=  [hits]

    last_stars_list += [last_stars]
    last_title_list += [last_title]

  def _ids(row, name, name2):
    return [vocabs[name].id(x) for x in row[name2].split(',') if x]

  udf_['watch_vids'] = watch_vids_list   
  udf_['watch_times'] = watch_times_list
  udf_['cids'] = cids_list
  udf_['class_ids'] = class_ids_list
  udf_['second_classes'] = second_classes_list
  udf_['is_intacts'] = is_intacts_list

  udf_['titles'] = titles_list
  udf_['stars_list'] = stars_list
  udf_['all_stars_list'] = all_stars_list
  udf_['all_stars_str_list'] = all_stars_str_list
  udf_['first_stars_list'] = first_stars_list
  udf_['hits'] = hits_list
  udf_['durations'] = durs_list
  udf_['freshes'] = freshes_list

  udf_['last_stars'] = last_stars_list
  udf_['last_title'] = last_title_list

  d[index] = udf_

def deal(index):
  ofile = f'{FLAGS.odir}/record_{index}.TMP'

  if not FLAGS.force:
    if glob.glob(f'{FLAGS.odir}/record_{index}*'):
      print(f'{FLAGS.odir}/record_{index} exists')
      return

  num_records = 0

  total = len(df)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  df_ = df.iloc[start:end]

  def _id(row, name):
    return vocabs[name].id(row[name])

  def _id2(row, name):
    if not np.isnan(row[name]):
      return vocabs[name].id(int(row[name]))
    else:
      return vocabs[name].unk_id()

  missing_image_emb = list(np.random.uniform(-0.05, 0.05,(128,)))
  with melt.tfrecords.Writer(ofile) as writer:
    for _, row in tqdm(df_.iterrows(), total=len(df_), ascii=True):      
      feature = {}

      did = row['did']
      vid = row['vid']

      day = 0 if not FLAGS.mark == 'train' else FLAGS.day
      feature['day'] = day
      feature['index'] = row['index']

      feature['label'] = row['label'] if 'label' in row else 0

      # -------id 
      feature['id'] = f'{did}\t{vid}'
      feature['did'] = _id(row, 'did')
      feature['vid'] = _id(row, 'vid')

      feature['did_'] = did
      feature['vid_'] = vid

      # -------user
      feature['watch_times'] = gezi.pad(row['watch_times'], 50)
      feature['watch_vids'] = gezi.pad(row['watch_vids'], 50)
      feature['cids'] = gezi.pad(row['cids'], 50)
      feature['class_ids'] = gezi.pad(row['class_ids'], 50)
      feature['second_classes'] = gezi.pad(row['second_classes'], 50)
      feature['is_intacts'] = gezi.pad(row['is_intacts'], 50)

      feature['stars_list'] = row['stars_list']
      feature['all_stars_list'] = row['all_stars_list']
      feature['first_stars_list'] = gezi.pad(row['first_stars_list'], 50)
      feature['titles'] = row['titles']
      feature['durations'] = gezi.pad([float(x) for x in row['durations']], 50, 0.)
      feature['freshes'] = gezi.pad(row['freshes'], 50)
      feature['hits'] = row['hits']

      feature['last_title'] = row['last_title']
      feature['last_stars'] = row['last_stars']

      feature['num_shows'] = row['num_shows']

      show_vids = json.loads(row['vids'])
      show_vids = [x for x in show_vids if x != vid]
      np.random.shuffle(show_vids)
      feature['show_vids'] = gezi.pad(show_vids, 50)
  
      # -------item
      # 视频所属合集 
      feature['cid'] = _id(row, 'cid')
      feature['cid_rate'] = float(row['cid_rate'])
      # 视频类别
      feature['class_id'] = _id(row, 'class_id')
      # 二级分类,脱敏id值,可能为空(0),比如 电影下的战争片
      feature['second_class'] = _id(row, 'second_class')
      # 正短片类型，比如正片，短片、预告片、花絮
      feature['is_intact'] = _id(row, 'is_intact')

      # 视频长度
      feature['duration'] = row['duration']
      # 归一化 0-9
      feature['duration_'] = int(row['duration_'])

      # 视频在芒果tv 全站最近n日的播放量，只具有横向比较意义
      feature['vv'] = row['vv']
      feature['vv_'] = int(row['vv_'])

      feature['ctr'] = float(row['ctr'])
      feature['ctr_'] = int(row['ctr_'])

      feature['title_length'] = row['title_length']
      feature['title_length_'] = int(row['title_length_'])

      feature['vtimestamp'] = row['vtimestamp']
      feature['fresh'] = row['timestamp'] - row['vtimestamp']

      # array
      feature['stars'] = [vocabs['stars'].id(x) for x in row['stars']]

      # -------context
      # 预览片id 预览片为0表示短片无预览 1表示长片对应的预览片 但是注意预览片id 可能在vid词典里面没有 近期样本里面没出现 所以设定 0 而不能是UNK=1
      feature['prev'] = vocabs['vid'].id(row['prev']) if row['prev'] else 0
      feature['has_prev'] = int(row['prev'] != 0)

      # 手机型号
      feature['mod'] = _id(row, 'mod')
      # 手机厂商
      feature['mf'] = _id(row, 'mf')
      # 手机操作系统
      feature['sver'] = _id(row, 'sver')
      # 芒果app版本
      feature['aver'] = _id(row, 'aver')

      # 用户区域
      feature['region'] = _id(row, 'region')
      
      feature['timestamp'] = row['timestamp']
      feature['hour'] = time.localtime(row['timestamp']).tm_hour
      feature['weekday'] = datetime.fromtimestamp(row['timestamp']).weekday()

      # -------other
      # feature['image_emb'] = [float(x) if x else 0. for x in row['image_emb'].split(',')]
      # if not feature['image_emb']:
      #   feature['image_emb'] = missing_image_emb
   
      feature['title'] = [vocabs['words'].id(x) for x in row['title'].split(',')]
      feature['story'] = [vocabs['words'].id(x) for x in row['story'].split(',')]

      feature['prev_is_intact'] = vocabs['is_intact'].id(row['prev_is_intact'])
      feature['prev_duration'] = float(row['prev_duration'])
      feature['prev_title_length'] = int(row['prev_title_length'])
      feature['prev_ctr'] = float(row['ctr'])
      feature['prev_vv'] = int(row['prev_vv'])
      feature['prev_duration_'] = int(row['prev_duration_'])
      feature['prev_title_length_'] = int(row['prev_title_length_'])
      feature['prev_ctr_'] = int(row['ctr_'])
      feature['prev_vv_'] = int(row['prev_vv_'])

      # ---------人肉特征
      # context_feats = gen_context_feats(row)
      # for context, feat in context_feats.items():
      #   feature[context] = feat

      # item_feats = gen_item_feats(row)
      # for item, feat in item_feats.items():
      #   feature[item] = feat

      # cross_feats = gen_cross_feats(row)
      # for matching, feat in cross_feats.items():
      #   feature[matching] = feat
 
      for key in feature:
        if isinstance(feature[key], list or tuple) and not feature[key]:
          feature[key] = [0]
        try:
          feature[key] = melt.gen_feature(feature[key])
        except Exception:
          print(key, feature[key])
          exit(0)
      record = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(record)
      num_records += 1

  ofile2 = ofile.replace('.TMP', f'.{num_records}')
  os.system('mv %s %s' % (ofile, ofile2))

def get_new_date(data):
  tmp_timestamp_first = data.groupby(['did'])['timestamp'].min().reset_index()
  del data['timestamp']
  data = pd.merge(data,tmp_timestamp_first,on=['did'],how='left',copy=False)
  return data

def main(_):  
  global df, odir, vocabs, vinfos, udf, timestamp

  if FLAGS.day == 0 or FLAGS.day > 30:
    FLAGS.mark = 'eval'

  if FLAGS.toy:
    FLAGS.odir = FLAGS.odir.replace('tfrecords', 'tfrecords-toy')
    # FLAGS.num_records = 1

  if FLAGS.lm:
    FLAGS.odir = FLAGS.odir.replace('tfrecords', 'tfrecords-lm')

  vocab_names = [
                  'vid', 'words', 'stars', 'did', 'region', 'sver', 
                  'mod', 'mf', 'aver', 'is_intact', 'second_class', 'class_id', 'cid',
                ]
  for vocab_name in vocab_names:
    vocabs[vocab_name] = gezi.Vocab(f'../input/all/{vocab_name}.txt')

  print('----loading context')
  context_file = f'../input/{FLAGS.mark}/part_{FLAGS.day}/context.parquet' if FLAGS.mark == 'train' else f'../input/{FLAGS.mark}/context.parquet'
  df = gezi.read_parquet(context_file)
  df = get_new_date(df)
  if FLAGS.lm:
    df = df.groupby('did', as_index=False).first()
  if FLAGS.mark == 'train':
    df = df.sample(frac=1, random_state=FLAGS.seed_)

  if FLAGS.toy:
    df = df[:50000]

  print('-----loading item')
  item_ifile = f'../input/{FLAGS.mark}/part_{FLAGS.day}/item.parquet' if FLAGS.mark == 'train' else f'../input/{FLAGS.mark}/item.parquet'
  idf = gezi.read_parquet(item_ifile)
  w = pd.read_csv('../input/all/bins.csv')
  cols = ['title_length', 'duration', 'vv']
  for col in cols:
    idf[f'{col}_'] = pd.cut(idf[col], w[col].values, labels=range(10))
    idf[f'{col}_'] = idf[f'{col}_'].astype(int)
    idf[f'{col}_'] = idf[f'{col}_'].apply(lambda x: max(x + 1, 1))

  def _ctr(x):
    bins = list(map(float, range(100)))
    bins = [x * 0.01 for x in bins]
    for i in range(100):
      if x <= bins[i]:
        return i + 1
    return i + 1
  idf['ctr_'] = idf.ctr.apply(_ctr)

  idf = idf.rename(columns={'timestamp':'vtimestamp'})

  df = df.merge(idf, how="left", on="vid")

  d1 = df.groupby(['did', 'cid']).size().reset_index(name='did_cid_counts')
  d2 = df.groupby('did').size().reset_index(name='did_counts')
  d3 = d1.merge(d2, on='did', how='left')
  d3['cid_rate'] = d3.did_cid_counts / d3.did_counts

  df = df.merge(d3[['did', 'cid', 'cid_rate']], how="left", on=['did', 'cid'])

  print('-------merge prev')
  idf['prev'] = idf['vid']
  cols =  ['is_intact', 'duration', 'title_length', 'ctr', 'vv', 'duration_', 'title_length_', 'ctr_', 'vv_']
  cols2 = [f'prev_{x}' for x in cols]
  m = dict(zip(cols, cols2))
  idf = idf.rename(columns=m)
  idf = idf[['prev', *cols2]]
  df = df.merge(idf, how="left", on='prev')
  df = df.fillna(0)

  dir_name = f'{FLAGS.mark}/part_{FLAGS.day}' if FLAGS.mark == 'train' else FLAGS.mark

  print('--------loading raw')
  raw_file = f'../input/train/raw.parquet'
  rdf = gezi.read_parquet(raw_file)
  df = df.merge(rdf, how="left", on="vid")

  print('--------loading user')
  user_file = f'../input/{dir_name}/user.parquet' 
# if not os.path.exists(user_file.replace('.parquet', '.csv')):
  udf = gezi.read_parquet(user_file)
  udf.watch = udf.watch.apply(json.loads)
  
  print('---------loading vinfo and move to map')
  # 通过gen_video_info.py生成
  vinfo = pd.read_csv('../input/all/vinfo_static.csv')
  vinfo = vinfo.fillna(0.)

  for _, row in tqdm(vinfo.iterrows(), total=len(vinfo), ascii=True):
    vinfos[row['vid']] = row

  print('--------- merge user and vinfo')
  timestamp = df.timestamp.values[0]
  print('month and day:', get_year_mon_day(timestamp))
  
  with Pool(FLAGS.num_records) as p:
    p.map(merge_uv, range(FLAGS.num_records))
  udf = pd.concat([x for _, x in d.items()])

  if FLAGS.gen_stars_corpus:
    with open(f'../input/{dir_name}/stars_click.txt', 'w') as out:
      all_stars_str_list = udf.all_stars_str_list.values
      for all_stars in all_stars_str_list:
        print(' '.join(all_stars), file=out)
    exit(0)

  ushow_file = f'../input/{dir_name}/user_shows.csv' 
  ushow = pd.read_csv(ushow_file)

  udf = udf.merge(ushow, on='did', how='left')

  df = df.merge(udf, how="left", on="did")

  odir = f'{FLAGS.odir}/{FLAGS.mark}'
  if FLAGS.mark == 'train':
    odir = f'{odir}/{FLAGS.day}' 

  os.system(f'mkdir -p {odir}')

  FLAGS.odir = odir

  if FLAGS.num_records > 1:
    with Pool(FLAGS.num_records) as p:
      p.map(deal, range(FLAGS.num_records))
  else:
    deal(0)

if __name__ == '__main__':
  app.run(main)

