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

import tensorflow as tf
import transformers
from transformers import AutoTokenizer

import gezi, melt
from utils import *
from transform import transformer

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_integer('num_records', 5, '')
flags.DEFINE_string('mark', 'xlm', '')
flags.DEFINE_integer('sample', 0, '')
flags.DEFINE_string('pretrained', 'tf-xlm-roberta-large', '')
flags.DEFINE_integer('seed_', 1024, '')
flags.DEFINE_integer('max_len', 192, '')
flags.DEFINE_integer('last_tokens', 50, '')
flags.DEFINE_bool('padding', False, '')
flags.DEFINE_bool('valid_by_lang', False, '')
flags.DEFINE_bool('clean', False, '')

df = None
dfs = []
odir = None
tokenizer = None

def deal(index):
  if not dfs:
    total = len(df)
    start, end = gezi.get_fold(total, FLAGS.num_records, index)
    df_ = df.iloc[start:end]
  else:
    df_ = dfs[index]
  num_records = len(df_)
  
  ofile = f'{odir}/record_{index}.{num_records}'

  with melt.tfrecords.Writer(ofile) as writer:
    for _, row in tqdm(df_.iterrows(), total=num_records):
      feature = {}
      for colum in df.columns:
        if colum == 'comment_text': 
          text = row['comment_text']
          if FLAGS.clean:
            text = transformer(text)
          input_word_ids = regular_encode(text, tokenizer, FLAGS.max_len, FLAGS.last_tokens, FLAGS.padding)
          feature['input_word_ids'] = melt.gen_feature(input_word_ids, np.int64)
        else:
          try:
            feature[colum] = melt.gen_feature(row[colum], df[colum].dtype)
          except Exception:
            print(colum)
            exit(0)
      
      record = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(record)

def main(_):  
  global df, dfs, odir, tokenizer

  if FLAGS.pretrained == 'tf-xlm-mlm-17-1280':
    FLAGS.mark = FLAGS.mark.replace('xlm', 'xlm2')
  FLAGS.pretrained = '../input/' + FLAGS.pretrained
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained)

  ifile = sys.argv[1]
  ifile2 = sys.argv[2] if 'test' in ifile else None
  df = gen_df(ifile, ifile2)

  record_name = os.path.basename(ifile).rstrip('.csv') 
  if FLAGS.valid_by_lang:
    record_name = record_name.replace('validation', 'validation-bylang')
    FLAGS.num_records = 3
    dfs = [df[df.lang=='tr'], df[df.lang=='it'], df[df.lang=='es']]

  if not FLAGS.padding:
    if FLAGS.max_len:
      FLAGS.mark = f'{FLAGS.mark}-{FLAGS.max_len}'
  if FLAGS.clean:
    FLAGS.mark = f'{FLAGS.mark}-cleaned'
  odir = f'{FLAGS.odir}/{FLAGS.mark}/{record_name}' 

  os.system(f'mkdir -p {odir}')

  with Pool(FLAGS.num_records) as p:
    p.map(deal, range(FLAGS.num_records))


if __name__ == '__main__':
  app.run(main)

