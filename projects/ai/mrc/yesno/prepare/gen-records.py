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
sys.path.append('..')
import os

from absl import app, flags
FLAGS = flags.FLAGS

import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
from transformers import AutoTokenizer

import gezi
from gezi import pad
import melt

import tensorflow as tf

df = None
tokenizer = None

def encode(text, max_len=None, last_tokens=None, padding=False):
  word_ids = tokenizer.encode(
      text,
      return_attention_mask=False, 
      return_token_type_ids=False
  )

  if max_len:
    if len(word_ids) != max_len:
      if len(word_ids) > max_len:
        if last_tokens:
          word_ids = [*word_ids[:max_len - last_tokens], *word_ids[-last_tokens:]]
        else:
          word_ids = word_ids[:max_len]
      elif padding:
        word_ids = gezi.pad(word_ids, max_len)
  
  return word_ids

def build_features(index):
  df_ = gezi.get_df_fold(df, FLAGS.num_records, index)
  ofile = f'{FLAGS.out_dir}/{FLAGS.mark}/record_{index}.tfrec'
  excls = set(['paras', 'query', 'title'])
  with melt.tfrecords.Writer(ofile) as writer:
    for _, row in tqdm(df_.iterrows(), total=len(df_)):
      feature = {}
      for key in df_.columns:
        if key not in excls:
          feature[key] = row[key]
      try:
        feature['query'] = encode(row['query'], max_len=100, padding=FLAGS.padding)
        feature['title'] = encode(row['title'], max_len=100, padding=FLAGS.padding)
        feature['content'] = encode(''.join(row['paras']), max_len=FLAGS.max_len, last_tokens=50, padding=FLAGS.padding)
        head = encode(row['query'] + '[SEP]' + row['title'], max_len=200)
        content = encode(''.join(row['paras']), max_len=FLAGS.max_len - len(head), last_tokens=50, padding=FLAGS.padding)
        feature['all'] = [*head, *content]
        content2 = encode(''.join(row['paras']), max_len=FLAGS.max_len * 2, last_tokens=50, padding=FLAGS.padding)[-(FLAGS.max_len - len(head)):]
        feature['all2'] = [*head, *content2]
        # print(len(feature['all']), len(feature['all2']))
        # rationale_marks = [int(x) for x in json.loads(row['rationale_marks'])]
        try:
          rationale_marks = [int(x) for x in row['rationale_marks'].split(',')]
        except Exception:
          continue
        if np.sum(rationale_marks == 0):
          print(row['answer_type_mark'], rationale_marks)
          feature['answer_type_mark'] = 4
        feature['rationale_marks'] = [x + 1 for x in rationale_marks]
        if FLAGS.padding:
          feature['rationale_marks'] = gezi.pad(feature['rationale_marks'], 100)
        writer.write_feature(feature)
      except Exception:
        print(row)

def main(_):
  np.random.seed(FLAGS.seed_)

  assert FLAGS.padding
  # if FLAGS.padding:
  #   FLAGS.record_name += '-padded'
    
  FLAGS.out_dir += f'/{FLAGS.record_name}'
  if not os.path.exists(FLAGS.out_dir):
    print('make new dir: [%s]' % FLAGS.out_dir, file=sys.stderr)
    os.makedirs(FLAGS.out_dir)

  global df, tokenizer

  tokenizer = AutoTokenizer.from_pretrained(FLAGS.transformers_model)

  df = pd.read_csv(FLAGS.input)

  if 'train' in FLAGS.mark:
    print('df shuffle')
    df = df.sample(frac=1, random_state=FLAGS.seed_)
  
  # build_features(0)
  with Pool(FLAGS.num_records) as p:
    p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'train', 'train or dev')
  flags.DEFINE_integer('num_records', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('record_name', 'tfrecords', '')
  flags.DEFINE_string('transformers_model', 'bert-base-chinese', '')
  flags.DEFINE_integer('max_len', 512, '')
  flags.DEFINE_bool('padding', True, '')
  
  app.run(main) 
