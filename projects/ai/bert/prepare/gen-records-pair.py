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

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_integer('num_records', 5, '')
flags.DEFINE_string('mark', 'xlm-pair', '')
flags.DEFINE_integer('sample', 0, '')
flags.DEFINE_string('pretrained', 'tf-xlm-roberta-large', '')
flags.DEFINE_integer('seed_', 1024, '')
flags.DEFINE_integer('max_len', 192, '')
flags.DEFINE_integer('last_tokens', 50, '')
flags.DEFINE_bool('padding', True, '')
flags.DEFINE_bool('valid_by_lang', False, '')
flags.DEFINE_bool('concat_words', False, '')

df = None
df2 = None
dfs = []
dfs2 = []
odir = None
toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
langs = ['en', 'es', 'it', 'tr', 'fr', 'pu', 'ru']

tokenizer = None

def deal(index):
  if not dfs:
    total = len(df)
    start, end = gezi.get_fold(total, FLAGS.num_records, index)
    df_ = df.iloc[start:end]
    df2_ = df2.iloc[start:end]
  else:
    df_ = dfs[index]
    df2_ = dfs2[index]
  num_records = len(df_)
  ofile = f'{odir}/record_{index}.{num_records}'

  with melt.tfrecords.Writer(ofile) as writer:
    for i in tqdm(range(len(df_)), ascii=True):
      row = df_.iloc[i]
      row2 = df2_.iloc[i]
      assert row['id'] == row2['id']
      feature = {}
      for colum in df.columns:
        if colum == 'comment_text': 
          text = row['comment_text']
          text2 = row2['comment_text']
          if FLAGS.concat_words:
            input_word_ids = regular_encode(text, tokenizer, FLAGS.max_len, FLAGS.last_tokens, padding=False)
            input_word_ids2 = regular_encode(text2, tokenizer, FLAGS.max_len, FLAGS.last_tokens, padding=False)
            input_word_ids2[0] = 2
            input_word_ids = [*input_word_ids, *input_word_ids2]  
            if FLAGS.padding:
              input_word_ids = padding_words(input_word_ids, FLAGS.max_len * 2)
            feature['input_word_ids'] = melt.gen_feature(input_word_ids, np.int64)
          else:
            input_word_ids = regular_encode(text, tokenizer, FLAGS.max_len, FLAGS.last_tokens, padding=FLAGS.padding)
            input_word_ids2 = regular_encode(text2, tokenizer, FLAGS.max_len, FLAGS.last_tokens, padding=FLAGS.padding)
            feature['input_word_ids2'] = melt.gen_feature(input_word_ids, np.int64)
            feature['input_word_ids'] = melt.gen_feature(input_word_ids2, np.int64)
            assert(len(input_word_ids) == 192)
            assert(len(input_word_ids2) == 192)
          if FLAGS.padding:
            assert(len(input_word_ids) == FLAGS.max_len)
            assert(len(input_word_ids2) == FLAGS.max_len)
        else:
          try:
            feature[colum] = melt.gen_feature(row[colum], df[colum].dtype)
          except Exception:
            print(colum)
            exit(0)
          try:
            feature[f'{colum}2'] = melt.gen_feature(row[colum], df[colum].dtype)
          except Exception:
            print(colum)
            exit(0)
      
      record = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(record)

def main(_):  
  global df, df2, dfs, dfs2, odir, tokenizer

  if FLAGS.pretrained == 'tf-xlm-mlm-17-1280':
    FLAGS.mark = FLAGS.mark.replace('xlm', 'xlm2')
  FLAGS.pretrained = '../input/' + FLAGS.pretrained
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained)

  ifile = sys.argv[1]
  ifile2 = sys.argv[2]

  df = gen_df(ifile)
  df2 = gen_df(ifile2)

  print(len(df), len(df2))
  # exit(0)
  d_ = df2[~df2.id.isin(df.id.values)]
  if len(d_):
    ids = d_.id.values
    row = df.iloc[0]
    row['comment_text'] = 'null'
    for i , id_ in tqdm(enumerate(ids), total=len(ids), ascii=True):
      row['id'] = id_
      df = df.append(row.copy(), ignore_index=True)
    
    df = df.sort_values('id')
    df2 = df2.sort_values('id')

    df = shuffle(df, random_state=FLAGS.seed_)
    df2 = shuffle(df2, random_state=FLAGS.seed_)

  print(len(df), len(df2), len(df2[~df2.id.isin(df.id.values)]))

  record_name = os.path.basename(ifile).rstrip('.csv')
  if FLAGS.valid_by_lang:
    record_name = record_name.replace('validation', 'validation-bylang')
    FLAGS.num_records = 3
    dfs = [df[df.lang=='tr'], df[df.lang=='it'], df[df.lang=='es']] 
    dfs2 = [df2[df2.lang=='tr'], df2[df2.lang=='it'], df2[df2.lang=='es']] 

  if FLAGS.concat_words:
    FLAGS.mark = f'{FLAGS.mark}-concat'
  if not FLAGS.padding:
    if FLAGS.max_len:
      FLAGS.mark = f'{FLAGS.mark}-{FLAGS.max_len}'
  odir = f'{FLAGS.odir}/{FLAGS.mark}/{record_name}' 

  os.system(f'mkdir -p {odir}')

  with Pool(FLAGS.num_records) as p:
    p.map(deal, range(FLAGS.num_records))

if __name__ == '__main__':
  app.run(main)

