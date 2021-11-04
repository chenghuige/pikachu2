#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-tfrecord.py
#        \author   chenghuige  
#          \date   2019-09-11 11:00:01.818073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
from gezi.util import index
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import pandas as pd
import jieba

import melt as mt
import gezi
from gezi import tqdm

def main(_):
  word_counter = gezi.WordCounter()
  title_counter = gezi.WordCounter()
  asr_counter = gezi.WordCounter()
  
  record_files = []
  dirs = ['../input/pointwise', '../input/pairwise', '../input/test_a/', '../input/test_b']
  tdirs = tqdm(dirs)
  for dir in tdirs:
    ids = []
    tdirs.set_postfix({'dir': dir})
    record_files = gezi.list_files(f'{dir}/*.tfrecords')
    ic(record_files)
    t = tqdm(record_files)
    for record_file in t:
      t.set_postfix({'file': record_file})
      for i, item in enumerate(tf.data.TFRecordDataset(record_file)):
        x = mt.decode_example(item)
        title = gezi.decode(x['title'])[0]
        asr = gezi.decode(x['asr_text'])[0]
        title_words = list(jieba.cut(title))
        asr_words = list(jieba.cut(asr))
        if i == 0:
          ic(title, asr)
          ic(title_words, asr_words)
        word_counter.adds(title_words)
        word_counter.adds(asr_words)
        title_counter.adds(title_words)
        asr_counter.adds(asr_words)

  word_counter.save('../input/word_vocab.txt')
  title_counter.save('../input/title_vocab.txt')
  asr_counter.save('../input/asr_vocab.txt')

if __name__ == '__main__':
  app.run(main)  
  
