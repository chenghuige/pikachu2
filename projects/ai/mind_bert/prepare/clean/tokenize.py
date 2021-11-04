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

flags.DEFINE_string('odir', '../input/tfrecords', '')
flags.DEFINE_integer('num_records', 5, '')
flags.DEFINE_string('mark', 'xlm', '')
flags.DEFINE_integer('sample', 0, '')
flags.DEFINE_string('pretrained', '../input/tf-xlm-roberta-large', '')
flags.DEFINE_integer('seed_', 0, '')
flags.DEFINE_bool('padding', True, '')
flags.DEFINE_integer('max_len', 192, '')

df = None
odir = None
toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
tokenizer = None

def regular_encode(text, tokenizer):
  enc_di = tokenizer.encode_plus(
      text,
      return_attention_masks=False, 
      return_token_type_ids=False,
      pad_to_max_length=True,
      max_length=FLAGS.max_len
  )

  return np.array(enc_di['input_ids'])

m = Manager().dict()

def deal(index):
  total = len(df)
  start, end = gezi.get_fold(total, FLAGS.num_records, index)
  num_records = end - start
  df_ = df.iloc[start:end]

  for _, row in tqdm(df_.iterrows(), total=num_records):
    text = row['comment_text']
    for colum in df.columns:
      if colum == 'comment_text': 
        text = row['comment_text']
        input_word_ids = regular_encode(text, tokenizer)
      else:
        feature[colum] = melt.gen_feature(row[colum], df[colum].dtype)
      
def _float_cols(df, names):
  for name in names:
    if not name in df.columns:
      df[name] = 0.
    else:
      df[name] = df[name].astype(float)

def main(_):  
  global df, odir, tokenizer
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained)

  ifile = sys.argv[1]
  df = pd.read_csv(ifile)
  df = df.rename({'identity_threat': 'identity_hate', 'content': 'comment_text'}, axis=1)

  if 'translated' in df.columns:
    df['comment_text'] = df['translated']

  df = df[~df.comment_text.isnull()]

  if 'id' in df.columns:
    df['id'] = df['id'].astype(str)

  if 'unintend' in ifile:
    FLAGS.num_records = 40
    if FLAGS.sample == 1:
      df = pd.concat([df[df.toxic==0].sample(100000, random_state=FLAGS.seed_), df[df.toxic!=0]])
    elif FLAGS.sample == 2:
      df.toxic = df.toxic.round().astype(int)
      df = pd.concat([df[df.toxic==0].sample(100000, random_state=FLAGS.seed_), df[df.toxic==1]])

  if 'test' in ifile:
    FLAGS.num_records = 1
  else:
    df = shuffle(df, random_state=FLAGS.seed_)
    if 'valid' in ifile:
      FLAGS.num_records = 5
  
  _float_cols(df, toxic_types)

  record_name = os.path.basename(ifile).rstrip('.csv') 

  if not FLAGS.padding:
    FLAGS.mark = f'{FLAGS.mark}-nopad'
  odir = f'{FLAGS.odir}/{FLAGS.mark}/{record_name}' 

  os.system(f'mkdir -p {odir}')

  with Pool(FLAGS.num_records) as p:
    p.map(deal, range(FLAGS.num_records))


if __name__ == '__main__':
  app.run(main)

