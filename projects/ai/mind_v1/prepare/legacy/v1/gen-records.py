#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2019-07-27 22:33:36.314010
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app as absl_app
from absl import flags
FLAGS = flags.FLAGS

import glob
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
from collections import defaultdict
import numpy as np
import traceback
import json
from datetime import datetime

import gezi
from gezi import pad
import melt

import tensorflow as tf

from config import *

def to_datetime(s):
  return datetime.strptime(s,"%m/%d/%Y %I:%M:%S %p")

def to_timestamp(x):
  return int(datetime.timestamp(x))

df = None
uid_vocab, did_vocab = None, None
uid_vocab2, did_vocab2 = None, None
cat_vocab = None
scat_vocab = None
entity_vocab = None
entity_type_vocab = None
news_info = None
start_timestamps = {}

X = 1

behaviors_names = ['impression_id', 'uid', 'time', 'history', 'impressions']
news_names = ['did', 'cat', 'sub_cat', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

def build_features(index):
  total = len(df)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  df_ = df.iloc[start:end]

  num_records = 0

  buffer_size = None if (FLAGS.mark != 'train' or not FLAGS.shuffle_impressions) else FLAGS.shuffle_buffer_size
  ofile = f'{FLAGS.out_dir}/{FLAGS.mark}/record_{index}.TMP'
  folder_name = FLAGS.mark
  if FLAGS.neg_parts > 1:
    folder_name = f'{FLAGS.mark}-{FLAGS.neg_part}'
    os.system(f'mkdir -p {FLAGS.out_dir}/{folder_name}')
    ofile = f'{FLAGS.out_dir}/{FLAGS.mark}-{FLAGS.neg_part}/record_{index}.TMP'
  writer = melt.tfrecords.Writer(ofile, buffer_size=buffer_size) 

  if FLAGS.mark == 'train' and FLAGS.train_by_day:
    # 2019 11 9 -> 11 14
    num_days = 7
    num_records_list = [0] * num_days
    ofiles = []
    writers = []
    for i in range(num_days):
      os.system(f'mkdir -p {FLAGS.out_dir}/{folder_name}-days/{i}')
      ofiles += [f'{FLAGS.out_dir}/{FLAGS.mark}-days/{i}/record_{index}.TMP']
      writers += [melt.tfrecords.Writer(ofiles[-1], buffer_size=buffer_size)]

  for _, row in tqdm(df_.iterrows(), total=len(df_), ascii=True):
    time_ = row['time']
    day = int(time_.split()[0].split('/')[1]) - 9
    if FLAGS.day is not None and day != FLAGS.day:
      continue
    x = to_datetime(time_)
    weekday = x.weekday() 
    hour = x.hour
    timestamp = to_timestamp(x) 

    impressions = row['impressions'].split()
    impression_id = row['impression_id']
    uid = uid_vocab.id(row['uid'])

    try:
      history = [did_vocab.id(x) for x in reversed(row['history'].split())]
    except Exception:
      # print(row['history'], row['impression_id'])
      history = []
    
    feature = {}
    feature['uid_'] = row['uid']
    feature['uid'] = uid
    feature['day'] = day
    feature['weekday'] = weekday
    feature['hour'] = hour
    feature['history'] = history
    feature['impression_id'] = impression_id
    feature['uid_in_train'] = int(uid_vocab2.has(row['uid']))
    feature['impression_len'] = len(impressions) 

    # TODO need history entities per did 
    # feature['history_cats_'] = []
    # feature['history_sub_cats_'] = []
    feature['history_cats'] = []
    feature['history_sub_cats'] = []
    if not FLAGS.slim:
      feature['history_title_entities'] = []
      feature['history_title_entity_types'] = []
      feature['history_abstract_entities'] = []
      feature['history_abstract_entity_types'] = []
    for did in history:
      did = did_vocab.key(did)
      news = news_info[did]
      # feature['history_cats_'].append(news['cat'])
      # feature['history_sub_cats_'].append(news['sub_cat'])
      feature['history_cats'].append(cat_vocab.id(news['cat']))
      feature['history_sub_cats'].append(scat_vocab.id(news['sub_cat']))
      if not FLAGS.slim:
        try:
          title_entities = json.loads(news['title_entities'])
          l, l2 = [], []
          for m in title_entities:
            entity = m['WikidataId']
            l.append(entity_vocab.id(entity))
            l2.append(entity_type_vocab.id(m['Type']))
          l = pad(l, FLAGS.max_his_title_entities, X)
          l2 = pad(l2, FLAGS.max_his_title_entities, X)
          feature['history_title_entities'] += l
          feature['history_title_entity_types'] += l2
        except Exception:
          feature['history_title_entities'] += [X] * FLAGS.max_his_title_entities 
          feature['history_title_entity_types'] += [X] * FLAGS.max_his_title_entities

        try:
          abstract_entities = json.loads(news['abstract_entities'])
          l, l2 = [], []
          for m in title_entities:
            entity = m['WikidataId']
            l.append(entity_vocab.id(entity))
            l2.append(entity_type_vocab.id(m['Type']))
          l = pad(l, FLAGS.max_his_abstract_entities, X)
          l2 = pad(l2, FLAGS.max_his_abstract_entities, X)
          feature['history_abstract_entities'] += l
          feature['history_abstract_entity_types'] += l2
        except Exception:
          feature['history_abstract_entities'] += [X] * FLAGS.max_his_abstract_entities
        feature['history_abstract_entity_types'] += [X] * FLAGS.max_his_abstract_entities

    feature['hist_len'] = len(feature['history'])
    if FLAGS.record_padded:
      for key in ['history', 'history_cats', 'history_sub_cats']:
        feature[key] = pad(feature[key], FLAGS.max_history)
      if not FLAGS.slim:
        feature['history_title_entities'] = pad(feature['history_title_entites'], FLAGS.max_history * FLAGS.max_his_title_entities)
        feature['history_title_entity_types'] = gezi.pad(feature['history_title_entity_types'], FLAGS.max_history * FLAGS.max_his_title_entities)
        feature['history_abstract_entities'] = pad(feature['history_abstract_entities'], FLAGS.max_history * FLAGS.max_his_abstract_entities)
        feature['history_abstract_entitiy_types'] = pad(feature['history_abstract_entitity_types'], FLAGS.max_history * FLAGS.max_his_abstract_entities)

    if FLAGS.neg_parts > 1:
      indexes = list(range(len(impressions)))
      np.random.shuffle(indexes)

    for i, impression in enumerate(impressions):
      if '-' in impression:
        did_, click = impression.split('-')  
      else:
        did_, click = impression, '0'
      click = int(click)

      if FLAGS.neg_parts > 1:        
        if not click and indexes[i] % FLAGS.neg_parts != FLAGS.neg_part:
          continue

      start_timestamp = start_timestamps[did_]
      fresh = timestamp - start_timestamp
      did = did_vocab.id(did_)

      feature['fresh'] = fresh
      feature['did_in_train'] = int(did_vocab2.has(did_))

      feature['click'] = click
      feature['did_'] = did_
      feature['did'] = did
      feature['id'] = impression_id * 100 + i
      feature['position'] = i

      news = news_info[did_]

      # feature['cat_'] = news['cat']
      # feature['sub_cat_'] = news['sub_cat']
      feature['cat'] = cat_vocab.id(news['cat'])
      feature['sub_cat'] = scat_vocab.id(news['sub_cat'])
      feature['title_len'] = len(news['title'].split())
      try:
        feature['abstract_len'] = len(news['abstract'].split())
      except Exception:
        # Nan
        feature['abstract_len'] = 0

      feature['title_entities'] = []
      feature['title_entity_types'] = []
      feature['abstract_entities'] = []
      feature['abstract_entity_types'] = []

      try:
        title_entities = json.loads(news['title_entities'])
        for m in title_entities:
          entity = m['WikidataId']
          feature['title_entities'].append(entity_vocab.id(entity))
          feature['title_entity_types'].append(entity_type_vocab.id(m['Type']))
      except Exception:
        pass

      try:
        abstract_entities = json.loads(news['abstract_entities'])
        for m in title_entities:
          entity = m['WikidataId']
          feature['abstract_entities'].append(entity_vocab.id(entity))
          feature['abstract_entity_types'].append(entity_type_vocab.id(m['Type']))
      except Exception:
        pass

      if FLAGS.record_padded:
        for key in ['title_entities', 'title_entity_types']:
          feature[key] = pad(feature[key], FLAGS.max_title_entities)

        for key in ['abstract_entities', 'abstract_entity_types']:
          feature[key] = pad(feature[key], FLAGS.max_abstract_entities)      

      feature_ = {}
      for key in feature:
        feature_[key] = feature[key]
        if isinstance(feature[key], list or tuple) and not feature[key]:
          feature_[key] = [X]
      for key in feature_:
        try:
          feature_[key] = melt.gen_feature(feature_[key])
        except Exception:
          print(key, feature[key])
          print(traceback.format_exc())
          exit(0)

      record = tf.train.Example(features=tf.train.Features(feature=feature_))

      if FLAGS.mark == 'train' and FLAGS.train_by_day:
        writer = writers[day]

      writer.write(record)

      if FLAGS.mark == 'train' and FLAGS.train_by_day:
        num_records_list[day] += 1
      else:
        num_records += 1

  if FLAGS.mark == 'train' and FLAGS.train_by_day:
    for i in range(num_days):
      writers[i].close()   
      if num_records_list[i] == 0:
        os.system('rm -rf %s' % ofiles[i])
      else:
        ofile2 = ofiles[i].replace('.TMP', f'.{num_records_list[i]}')
        os.system('mv %s %s' % (ofiles[i], ofile2))
  else:
    writer.close()
    if num_records == 0:
      os.system('rm -rf %s' % ofile)
    else:
      ofile2 = ofile.replace('.TMP', f'.{num_records}')
      os.system('mv %s %s' % (ofile, ofile2))


def main(_):
  np.random.seed(FLAGS.seed_)

  files = gezi.list_files(FLAGS.in_dir)
  print('input', FLAGS.in_dir)

  FLAGS.out_dir += f'/{FLAGS.record_name}'
  if not os.path.exists(FLAGS.out_dir):
    print('make new dir: [%s]' % FLAGS.out_dir, file=sys.stderr)
    os.makedirs(FLAGS.out_dir)

  if FLAGS.train_by_day and FLAGS.shuffle_impressions:
    assert FLAGS.day is not None

  global df, uid_vocab, did_vocab, uid_vocab2, did_vocab2
  global cat_vocab, scat_vocab, entity_vocab, entity_type_vocab
  behaviors_file = f'{FLAGS.in_dir}/{FLAGS.mark}/behaviors.tsv'
  if FLAGS.mark == 'train' and FLAGS.day == 6:
    behaviors_file = f'{FLAGS.in_dir}/dev/behaviors.tsv'
  print('behaviors_file', behaviors_file)
  df = pd.read_csv(behaviors_file, sep='\t', names=behaviors_names)
  if FLAGS.mark == 'train':
    print('behaviors_df shuffle')
    df = df.sample(frac=1, random_state=FLAGS.seed_)
  uid_vocab = gezi.Vocab(f'{FLAGS.in_dir}/uid.txt')
  did_vocab = gezi.Vocab(f'{FLAGS.in_dir}/did.txt')
  uid_vocab2 = gezi.Vocab(f'{FLAGS.in_dir}/train/uid.txt')
  did_vocab2 = gezi.Vocab(f'{FLAGS.in_dir}/train/did.txt')
  cat_vocab = gezi.Vocab(f'{FLAGS.in_dir}/cat.txt')
  scat_vocab = gezi.Vocab(f'{FLAGS.in_dir}/sub_cat.txt')
  entity_vocab = gezi.Vocab(f'{FLAGS.in_dir}/entity.txt')
  entity_type_vocab = gezi.Vocab(f'{FLAGS.in_dir}/entity_type.txt')

  for line in open(f'{FLAGS.in_dir}/start_times.txt'):
    did, timestamp, _ = line.strip().split('\t')
    start_timestamps[did] = int(timestamp)

  global news_info
  # ndf = pd.read_csv(f'{FLAGS.in_dir}/{FLAGS.mark}/news.tsv', sep='\t', names=news_names)
  news_info = {}
  # for _, row in tqdm(ndf.iterrows(), total=len(ndf), ascii=True, desc='news_info'):
  #   news_info[row['did']] = row
  news_file = f'{FLAGS.in_dir}/{FLAGS.mark}/news.tsv'
  if FLAGS.mark == 'train' and FLAGS.day == 6:
    news_file = f'{FLAGS.in_dir}/dev/news.tsv'
  total = len(open(news_file).readlines())
  for line in tqdm(open(news_file), total=total, ascii=True, desc='news_info'):
    l = line.strip('\n').split('\t')
    m = {}
    for i, name in enumerate(news_names):
      m[name] = l[i]
    news_info[l[0]] = m

  with Pool(FLAGS.num_records) as p:
    p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'train', 'train or dev')
  flags.DEFINE_integer('num_records', 40, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_bool('shuffle_impressions', False, '')
  flags.DEFINE_bool('train_by_day', False, '')
  flags.DEFINE_string('record_name', 'tfrecords', '')
  flags.DEFINE_integer('day', None, '')
  flags.DEFINE_integer('shuffle_buffer_size', 100000, '')
  flags.DEFINE_bool('slim', False, '')
  flags.DEFINE_integer('neg_parts', 1, '')
  flags.DEFINE_integer('neg_part', 0, '')
  
  absl_app.run(main) 
