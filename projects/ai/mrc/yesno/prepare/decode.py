#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   decode.py
#        \author   chenghuige  
#          \date   2021-01-10 14:48:28.462282
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_list('input', '', '')
flags.DEFINE_string('output', '', '')
flags.DEFINE_string('mark', None, '')

import sys 
import os
import pandas as pd
# import pandas_profiling as pp
import gezi
from gezi import tqdm
import sentencepiece as SP
sp = None

def decode(ids):
  li = []
  l = []
  vocab_size = sp.get_piece_size()
  for id_ in ids:
    if id_ in [1, -1, vocab_size]:
      l = sp.DecodeIds(l)
      li.append(l)
      l = []
    else:
      l.append(id_)
  return li

def main(argv):
  assert FLAGS.mark in ['train', 'dev']
  files = FLAGS.input
  l = []
  for file in tqdm(files, desc='load_pickle'):
    d = gezi.load_pickle(file)
    l.extend(d)

  print('total:', len(l))

  global sp
  sp = SP.SentencePieceProcessor()
  sp.load('../input/resource/bpe.50000.new.model')

  res = []
  # [one_sample, sent_pieces_list, rationale_mark_list, answer_type_mark, query]
  for items in tqdm(l, desc='decode'):
    if len(items) == 6:
      one_sample, _, rationale_marks, answer_type_mark, query, title = items
    else:
      one_sample, _, rationale_marks, answer_type_mark, query = items
    li = decode(one_sample)
    query_ = li[0]
    title = li[1]
    paras = li[2:]
    assert len(paras) == len(rationale_marks)
    # if query_ == '事业工勤人员不能当第一书记吗':
    #   print(items)
    #   print(rationale_marks)
    #   print(','.join(map(str, rationale_marks)))
    rationale_marks = ','.join(map(str, rationale_marks))
    # print(query, answer_type_mark, title, list(zip(paras, rationale_marks)))

    res.append([query_, answer_type_mark, title, paras, rationale_marks])

  df = pd.DataFrame(res, columns=['query', 'answer_type_mark', 'title', 'paras', 'rationale_marks'])

  df['num_chars'] = df.paras.apply(lambda x: sum([len(u) for u in x]))
  # report = pp.ProfileReport(df)
  # print(report)
  df.to_csv(f'../input/{FLAGS.mark}.csv', index=False)

if __name__ == '__main__':
  app.run(main)    
