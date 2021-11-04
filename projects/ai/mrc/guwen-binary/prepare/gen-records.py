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

import gezi
from gezi import pad
import melt

import tensorflow as tf
import transformers
from transformers import AutoTokenizer

m, tokenizer = None, None

PADDING_ID = 0

def padding_words(word_ids, max_len):
  if len(word_ids) < max_len:
    word_ids = word_ids + [PADDING_ID] * (max_len - len(word_ids))
  return word_ids

def regular_encode(text, tokenizer, max_len, last_tokens, padding):
  word_ids = tokenizer.encode(text)

  if len(word_ids) != max_len:
    if len(word_ids) > max_len:
      word_ids = [*word_ids[:max_len - last_tokens], *word_ids[-last_tokens:]]
    elif padding:
      word_ids = padding_words(word_ids, max_len)
  
  return word_ids

def build_features(index):
  start, end = gezi.get_fold(len(m), FLAGS.num_records, index)
  data = m[start: end]
  ofile = f'{FLAGS.out_dir}/{FLAGS.mark}/record_{index}.tfrec'
  total = len(data)
  with melt.tfrecords.Writer(ofile) as writer:
    for inst in tqdm(data, total=total):
      feature = {}
      feature['content'] = inst['Content']
      feature['did'] = inst['ID']
      feature['clen'] = len(feature['content'])
      for question in inst['Questions']:
        feature['question'] = question['Question']
        feature['answer'] = question['Answer'] if 'Answer' in question else 'E'
        feature['id'] = int(question['Q_id'])
        choices = question['Choices']

        question_ = tokenizer.encode(feature['question'])
        for choice in choices:
          feature['choice'] = choice[2:]
          choice_ = tokenizer.encode(feature['choice'])
          text_a = question_ + choice_[1:]
          text_b = regular_encode(feature['content'], tokenizer, FLAGS.max_len - len(text_a) + 1, 20, True)
          text = text_a + text_b[1:]
          feature['input_ids'] = text
          # feature['segment_ids'] = [0] * len(text_a) + [1] * len(text_b[1:])
          feature['segment_ids'] = [0] * len(question_) + [1] * len(choice_[1:]) + [1] * len(text_b[1:])
          feature['label'] = choice.startswith(feature['answer']) 
          if FLAGS.debug:
            print(feature)
            print(tokenizer.convert_ids_to_tokens(text))
          writer.write_feature(feature)

def main(_):
  global m, tokenizer

  np.random.seed(FLAGS.seed_)

  FLAGS.out_dir = f'{FLAGS.out_dir}/{FLAGS.record_name}'
  gezi.try_mkdir(f'{FLAGS.out_dir}/{FLAGS.mark}')

  m = gezi.read_json(f'{FLAGS.in_dir}/{FLAGS.ifile}')
  if FLAGS.mark == 'train':
    np.random.shuffle(m)
  tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

  if FLAGS.debug:
    build_features(0)
  else:
    with Pool(FLAGS.num_records) as p:
      p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('ifile', 'train.json', '')
  flags.DEFINE_string('in_dir', '../input/public', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'train', 'train or test')
  flags.DEFINE_integer('num_records', 10, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('record_name', 'tfrecords', '')
  flags.DEFINE_integer('max_len', 512, '')
  
  app.run(main) 
