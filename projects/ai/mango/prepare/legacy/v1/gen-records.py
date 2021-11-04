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

import gezi, melt
import tensorflow as tf
from gezi import hash_int64 as hash
logging = gezi.logging

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_string('mark', 'train', 'train or eval')
flags.DEFINE_integer('num_records', 32, '')
flags.DEFINE_integer('day', 30, '')
flags.DEFINE_bool('toy', False, '')
flags.DEFINE_integer('seed_', 12345, '')
flags.DEFINE_bool('force', True, '')
flags.DEFINE_bool('lm', False, '')

df = None
odir = None
vocabs = {}
vinfo = {}

def get_day(timestamp):
  x = time.localtime(timestamp)
  return x.tm_mday

def get_mon_day(timestamp):
  x = time.localtime(timestamp)
  return x.tm_mon, x.tm_mday

# 时间穿越或者当天的历史去掉 测试集合没有当天历史
def is_badtime(x, timestamp):
  if x >= timestamp:
    return True
  if get_mon_day(x) == get_mon_day(timestamp):
    return True

context_cols = ['prev', 'mod', 'mf', 'aver', 'sver', 'region']
item_cols = ['vid', 'duration_', 'title_length_', 'class_id', 'second_class', 'is_intact', 'vv_', 'ctr_']

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
def gen_match_feats(row):
  feats = {}
  for context_col in context_cols:
    for item_col in item_cols:
      feats[f'cross_{context_col}_{item_col}'] = hash(f'{row[context_col]}_{row[item_col]}')

  # l = []
  # for context_col in context_cols:
  #   for star in row['stars']:
  #     l += hash(f'{row[context_col]_{star}}')
  # feats['match_stars'] = l
  return feats

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

  if FLAGS.toy:
    df_ = df_[:2000]

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
      watch_times = []
      watch_vids = []
      cids = []
      class_ids = []
      second_classes = []
      is_intacts = []
      stars_list = []
      titles = []
      durs = []
      freshes = []

      match_stars = 0
      match_cids = 0
      match_class_ids = 0
      match_second_classes = 0
      match_is_intacts = 0
      match_prev = 0
      match_first_word = 0

      match_last_stars = 0
      match_last_cids = 0
      match_last_class_ids = 0
      match_last_second_classes = 0
      match_last_is_intacts = 0
      match_last_prev = 0
      match_last_first_word = 0

      cur_stars = set(row['stars'])

      index = 0
      for x in row['watch']:
        # 非常重要 否则数据穿越。。。
        if not is_badtime(x[0], row['timestamp']):
          index += 1
          wtime, wvid = x[0], x[1]
          watch_times += [wtime]
          watch_vids += [vocabs['vid'].id(wvid)]
          vrow = vinfo[wvid] if wvid in vinfo else None
          if vrow is not None:
            try:
              match_prev += int(row['prev'] == vrow['vid'])
              if index == 1:
                match_last_prev = int(row['prev'] == vrow['vid'])
              cids += [_id2(vrow, 'cid')]
              match_cids += int(row['cid'] == vrow['cid'])
              if index == 1:
                match_last_cids = int(row['cid'] == vrow['cid'])
              class_ids += [_id2(vrow, 'class_id')]
              match_class_ids = int(row['class_id'] == vrow['class_id'])
              if index == 1:
                match_last_class_ids = int(row['class_id'] == vrow['class_id'])
              second_classes += [_id2(vrow, 'second_class')]
              match_second_classes += int(row['second_class'] == vrow['second_class'])
              if index == 1:
                match_last_second_classes = int(row['second_class'] == vrow['second_class'])
              is_intacts += [_id2(vrow, 'is_intact')]
              match_is_intacts += int(row['is_intact'] == vrow['is_intact'])
              if index == 1:
                match_last_is_intacts = int(row['is_intact'] == vrow['is_intact'])
              durs += [float(vrow['duration'])]
              words = vrow['title'].split(',')[:10]
              if words and row['title'].split(','):
                x = int(row['title'].split(',')[0] == words[0])
                match_first_word += x
                if index == 1:
                  match_last_is_first_word = x
              words_ = [vocabs['words'].id(word) for word in words]
              titles += words_[:10]
              if not isinstance(vrow['stars'], list) and not isinstance(vrow['stars'], np.ndarray):
                vrow_stars = [vrow['stars']]
              else:
                vrow_stars = vrow['stars']
              for star in vrow_stars:
                if star in cur_stars:
                  match_stars += 1
                  if index == 1:
                    match_first_stars = 1
                  break
              vrow_stars = vrow_stars[:10]
              stars_ = [vocabs['stars'].id(star) for star in vrow_stars]
              stars_list += stars_[:10]
              freshes += [float(wtime - vrow['timestamp'])] 
            except Exception:
              print(vrow)
              print(x[1])
              print(traceback.format_exc())
              exit(0)
          else:
            cids += [1]
            class_ids += [1]
            second_classes += [1]
            is_intacts += [1]
            durs += [0.]

      if not durs:
        durs = [0.]
      if not freshes:
        freshes = [0.]

      feature['watch_times'] = watch_times
      feature['watch_vids'] = watch_vids
      feature['cids'] = cids
      feature['class_ids'] = class_ids
      feature['second_classes'] = second_classes
      feature['is_intacts'] = is_intacts
      feature['stars_list'] = stars_list
      feature['titles'] = titles
      feature['durations'] = durs
      feature['freshes'] = freshes

      # matching
      feature['match_stars'] = match_stars
      feature['match_cids'] = match_cids
      feature['match_class_ids'] = match_class_ids
      # TODO should be classes
      feature['match_second_classes'] = match_second_classes 
      feature['match_is_intacts'] = match_is_intacts
      feature['match_prev'] = match_prev
      feature['match_first_word'] = match_first_word
 
      feature['match_last_stars'] = match_last_stars
      feature['match_last_cids'] = match_last_cids
      feature['match_last_class_ids'] = match_last_class_ids
      feature['match_last_second_classes'] = match_last_second_classes
      feature['match_last_is_intacts'] = match_last_is_intacts
      feature['match_last_prev'] = match_last_prev
      feature['match_last_first_word'] = match_last_first_word

      # -------item
      # 视频所属合集 
      feature['cid'] = _id(row, 'cid')
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

      # ---------人肉特征
      context_feats = gen_context_feats(row)
      for context, feat in context_feats.items():
        feature[context] = feat

      item_feats = gen_item_feats(row)
      for item, feat in item_feats.items():
        feature[item] = feat

      match_feats = gen_match_feats(row)
      for matching, feat in match_feats.items():
        feature[matching] = feat
 
      for key in feature:
        if isinstance(feature[key], list or tuple) and not feature[key]:
          feature[key] = [0]
        feature[key] = melt.gen_feature(feature[key])
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
  global df, odir, vocabs, vinfo

  if FLAGS.day == 0 or FLAGS.day > 30:
    FLAGS.mark = 'eval'

  if FLAGS.toy:
    FLAGS.odir = FLAGS.odir.replace('tfrecords', 'tfrecords-toy')
    FLAGS.num_records = 1

  if FLAGS.lm:
    FLAGS.odir = FLAGS.odir.replace('tfrecords', 'tfrecords-lm')

  vocab_names = [
                  'vid', 'words', 'stars', 'did', 'region', 'sver', 
                  'mod', 'mf', 'aver', 'is_intact', 'second_class', 'class_id', 'cid',
                ]
  for vocab_name in vocab_names:
    vocabs[vocab_name] = gezi.Vocab(f'../input/all/{vocab_name}.txt')

  ifile = f'../input/{FLAGS.mark}/part_{FLAGS.day}/context.parquet' if FLAGS.mark == 'train' else f'../input/{FLAGS.mark}/context.parquet'
  df = gezi.read_parquet(ifile)
  df = get_new_date(df)
  if FLAGS.lm:
    df = df.groupby('did', as_index=False).first()
  if FLAGS.mark == 'train':
    df = df.sample(frac=1, random_state=FLAGS.seed_)

  ifile2 = f'../input/{FLAGS.mark}/part_{FLAGS.day}/user.parquet' if FLAGS.mark == 'train' else f'../input/{FLAGS.mark}/user.parquet'
  udf = gezi.read_parquet(ifile2)
  udf.watch = udf.watch.apply(json.loads)

  df = df.merge(udf, how="left", on="did")

  ifile3 = f'../input/{FLAGS.mark}/part_{FLAGS.day}/item.parquet' if FLAGS.mark == 'train' else f'../input/{FLAGS.mark}/item.parquet'
  idf = gezi.read_parquet(ifile3)
  w = pd.read_csv('../input/all/bins.csv')
  cols = ['title_length', 'duration', 'vv']
  for col in cols:
    idf[f'{col}_'] = pd.cut(idf[col], w[col].values, labels=range(10))
    idf[f'{col}_'] = idf[f'{col}_'].astype(int)

  def _ctr(x):
    bins = list(map(float, range(100)))
    bins = [x * 0.01 for x in bins]
    for i in range(100):
      if x <= bins[i]:
        return i
    return i
  idf['ctr_'] = idf.ctr.apply(_ctr)
  
  idf = idf.rename(columns={'timestamp':'vtimestamp'})

  df = df.merge(idf, how="left", on="vid")

  ifile4 = f'../input/train/raw.parquet'
  rdf = gezi.read_parquet(ifile4)

  df = df.merge(rdf, how="left", on="vid")

  # dvinfo = pd.read_csv('../input/all/vinfo_static.csv')
  # dvinfo['stars'] = dvinfo['stars'].apply(ast.literal_eval)
  # for _, row in tqdm(dvinfo.iterrows(), total=len(dvinfo), ascii=True): 
  #   vinfo[row['vid']] = row
  with gezi.Timer('loading vinfo', print_before=True, print_fn=print) as t:
    vinfo = pickle.load(open('../input/all/vinfos.pkl', 'rb'))

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

