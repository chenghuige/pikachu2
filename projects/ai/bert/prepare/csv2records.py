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
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm 

import tensorflow as tf

import gezi, melt

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_integer('num_records', 5, '')

df = None
odir = None

def deal(index):
  total = len(df)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  num_records = end - start
  df_ = df.iloc[start:end]
  ofile = f'{odir}/record_{index}.{num_records}'

  columns = df_.columns
  with melt.tfrecords.Writer(ofile, 1000) as writer:
    for _, row in tqdm(df_.iterrows(), total=num_records):
      feature = {}
      for colum in columns:
        feature[colum] = melt.gen_feature(row[colum], df[colum].dtype)
      
      record = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(record)
    

def main(_):  
  global df, odir
  ifile = sys.argv[1]
  df = pd.read_csv(ifile)

  if 'comment_text' in df.columns:
    del df['comment_text']

  if not 'toxic' in df.columns:
    df['toxic'] = 0.
  else:
    df['toxic'] = df['toxic'].astype(float)

  if 'id' in df.columns:
    df['id'] = df['id'].astype(str)

  if 'unintend' in ifile:
    # df = pd.concat([df[df.toxic==0].sample(100000), df[df.toxic!=0]])
    df = pd.concat([df[df.toxic==0].sample(100000), df[df.toxic==1]])
    FLAGS.num_records = 40

  if 'test' in ifile:
    FLAGS.num_records = 1

  record_name = os.path.basename(ifile).rstrip('.csv') 
  odir = f'{FLAGS.odir}/{record_name}' 

  os.system(f'mkdir -p {odir}')

  with Pool(FLAGS.num_records) as p:
    p.map(deal, range(FLAGS.num_records))


if __name__ == '__main__':
  app.run(main)

