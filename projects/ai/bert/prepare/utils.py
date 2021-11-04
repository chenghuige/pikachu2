#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   utils.py
#        \author   chenghuige  
#          \date   2020-04-23 11:44:13.836966
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from absl import flags
FLAGS = flags.FLAGS
import pandas as pd
from sklearn.utils import shuffle

toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
langs = ['en', 'es', 'it', 'tr', 'fr', 'pu', 'ru']
PADDING_ID = 1

def float_cols(df, names):
  for name in names:
    if not name in df.columns:
      df[name] = 0.
    else:
      df[name] = df[name].astype(float)

def get_src(ifile):
  if ifile.startswith('jigsaw-toxic-comment'):
    return 'toxic'
  elif ifile.startswith('jigsaw-unintended-bias'):
    return 'unintended'
  elif ifile.startswith('validation') or ifile.startswith('test'):
    return 'test'
  else:
    raise ValueError(ifile)

def get_lang(ifile):
  for lang in langs:
    if f'-{lang}' in ifile.replace('train', ''):
      return lang
  return 'en'

def get_trans(ifile):
  for lang in langs:
    if f'-{lang}' in ifile.replace('train', ''):
      return 1
  return 0
  
def gen_df(ifile, ifile2=None):
  ifile_name = os.path.basename(ifile)
  df = pd.read_csv(ifile)
  df = df.rename({'identity_threat': 'identity_hate', 'content': 'comment_text'}, axis=1)

  if 'translated' in df.columns:
    df['comment_text'] = df['translated']

  if ifile2:
    df2 = pd.read_csv(ifile2)
    df = pd.merge(df, df2, on='id')

  df = df[~df.comment_text.isnull()]

  if 'id' in df.columns:
    df['id'] = df['id'].astype(str)

  if 'lang' not in df.columns:
    df['lang'] = get_lang(ifile_name)

  df['src'] = get_src(ifile_name)
  df['trans'] = get_trans(ifile_name)

  if 'test' in ifile:
    FLAGS.num_records = 10
  else:
    df = shuffle(df, random_state=FLAGS.seed_)
    if 'valid' in ifile:
      FLAGS.num_records = 5

  float_cols(df, toxic_types)
  return df

def padding_words(word_ids, max_len):
  if len(word_ids) < max_len:
    word_ids = word_ids + [PADDING_ID] * (max_len - len(word_ids))
  return word_ids

def regular_encode(text, tokenizer, max_len, last_tokens, padding):
  word_ids = tokenizer.encode(
      text,
      return_attention_masks=False, 
      return_token_type_ids=False,
  )

  if len(word_ids) != max_len:
    if len(word_ids) > max_len:
      word_ids = [*word_ids[:max_len - last_tokens], *word_ids[-last_tokens:]]
    elif padding:
      word_ids = padding_words(word_ids, max_len)
  
  return word_ids
