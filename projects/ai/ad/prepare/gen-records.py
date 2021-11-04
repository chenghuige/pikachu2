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

import gezi, melt
import tensorflow as tf

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_string('mark', 'train', 'train or test')
flags.DEFINE_integer('num_records', 10, '')
flags.DEFINE_integer('min_count', 5, '')

df = None
adf = None
odir = None
cvocab = None
avocab = None

def deal_rows(rows):
  feature = {}
  uid = rows[0]['user_id']
  feature['id'] = melt.gen_feature([uid], np.int64)
  times, creative_ids, click_times = [], [], []
  # creative_id,ad_id,product_id,product_category,advertiser_id,industry
  ad_ids, product_ids, product_categories, advertiser_ids, industries = [], [], [], [], []
  
  for row in rows:
    times += [row['time']]
    creative_id = cvocab.id(str(row['creative_id']))
    creative_ids += [creative_id]
    click_times += [row['click_times']]
    ad_ids += [avocab.id(str(row['ad_id']))]
    product_ids += [row['product_id']]
    product_categories += [row['product_category']]
    advertiser_ids += [row['advertiser_id']]
    industries += [row['industry']]

  MAX_LEN = 128
  times = gezi.pad(times, MAX_LEN)
  creative_ids = gezi.pad(creative_ids, MAX_LEN)
  click_times = gezi.pad(click_times, MAX_LEN)
  ad_ids = gezi.pad(ad_ids, MAX_LEN)
  product_ids = gezi.pad(product_ids, MAX_LEN)
  advertiser_ids = gezi.pad(advertiser_ids, MAX_LEN) 
  product_categories = gezi.pad(product_categories, MAX_LEN)
  industries = gezi.pad(industries, MAX_LEN)

  feature['times'] = melt.gen_feature(times, np.int64)
  feature['creative_ids'] = melt.gen_feature(creative_ids, np.int64)
  feature['click_times'] = melt.gen_feature(click_times, np.int64)
  feature['ad_ids'] = melt.gen_feature(ad_ids, np.int64)
  feature['product_ids'] = melt.gen_feature(product_ids, np.int64)
  feature['product_categories'] = melt.gen_feature(product_categories, np.int64)
  feature['advertiser_ids'] = melt.gen_feature(advertiser_ids, np.int64)
  feature['industries'] = melt.gen_feature(industries, np.int64)

  if FLAGS.mark == 'train':
    feature['age'] = melt.gen_feature([row['age']], np.int64)
    feature['gender'] = melt.gen_feature(row['gender'], np.int64)
  else:
    feature['age'] = melt.gen_feature([0], np.int64)
    feature['gender'] = melt.gen_feature([0], np.int64)

  return feature

def deal(index):
  ofile = f'{odir}/record_{index}'

  num_records = 0
  rows = []
  pre_uid = None

  m = df.user_id % FLAGS.num_records == index
  d = df[m]
  d = d.sort_values(['user_id', 'time'])
  with melt.tfrecords.Writer(ofile) as writer:
    for _, row in tqdm(d.iterrows(), total=len(d)):
      feature = {}

      uid = row['user_id']
      # if uid % FLAGS.num_records != index:
      #   continue

      if uid != pre_uid:
        pre_uid = uid
        if rows:
          feature = deal_rows(rows)
          record = tf.train.Example(features=tf.train.Features(feature=feature))
          writer.write(record)
          num_records += 1
          rows = []
      rows += [row]
    
    if rows:
      feature = deal_rows(rows)
      record = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(record)
      num_records += 1
      
  ofile2 = ofile + f'.{num_records}'
  os.system('mv %s %s' % (ofile, ofile2))


def main(_):  
  global df, odir, avocab, cvocab

  ifile = f'../input/{FLAGS.mark}/click_log.csv'
  df = pd.read_csv(ifile)
  df = df.sample(frac=1)
  print(len(df))

  ifile2 = f'../input/{FLAGS.mark}/ad.csv'
  adf = pd.read_csv(ifile2)
  adf = adf.replace('\\N', 0)
  adf.product_id = adf.product_id.astype(int)
  adf.industry = adf.industry.astype(int)

  print(len(adf))

  df = df.merge(adf, how="left", on="creative_id")

  if FLAGS.mark == 'train':
    ifile3 = f'../input/{FLAGS.mark}/user.csv'
    udf = pd.read_csv(ifile3)
    df = df.merge(udf, how="left", on="user_id")

  # df = df.sort_values(['user_id', 'time'])

  odir = f'{FLAGS.odir}/{FLAGS.mark}' 

  os.system(f'mkdir -p {odir}')

  avocab = gezi.Vocab('../input/all/ad_ids.vocab', min_count=FLAGS.min_count)
  cvocab = gezi.Vocab('../input/all/creative_ids.vocab', min_count=FLAGS.min_count)

  with Pool(FLAGS.num_records) as p:
    p.map(deal, range(FLAGS.num_records))

if __name__ == '__main__':
  app.run(main)

