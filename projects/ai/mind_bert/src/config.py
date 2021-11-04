#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2020-04-15 10:31:47.629732
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os 

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import gezi

try:
  flags.DEFINE_string('model', None, '')
  flags.DEFINE_bool('multi_head', False, '')
  flags.DEFINE_string('pretrained', '../input/tf-xlm-roberta-large', '')
  flags.DEFINE_integer('max_len', None, 'xlm 192 bert 128')
  flags.DEFINE_bool('freeze_pretrained', False, '')
  flags.DEFINE_bool('valid_en', False, '')
  flags.DEFINE_alias('ve', 'valid_en')
  flags.DEFINE_bool('valid_bylang', False, '')
  flags.DEFINE_alias('vbl', 'valid_bylang')
  flags.DEFINE_bool('test_en', None, '')
  flags.DEFINE_bool('use_mlp', False, '')
  flags.DEFINE_bool('use_word_ids2', False, '')
  flags.DEFINE_bool('use_multi_dropout', False, '')
  flags.DEFINE_string('pooling', 'concat', '')
  flags.DEFINE_string('base_pooling', 'first', '')
  flags.DEFINE_bool('sample', False, '')
  flags.DEFINE_string('task', 'toxic', '')
  flags.DEFINE_float('dropout', 0.5, '')
  flags.DEFINE_float('sampling_rate', 1., '')
except Exception:
    pass

toxic_types = ['severe_toxic', 'obscene', 'identity_hate', 'threat', 'insult']
langs = ['en', 'es', 'it', 'tr', 'fr', 'pt', 'ru']
srcs = ['unintended', 'toxic', 'test']

BERT_GCS_PATH_SAVEDMODEL = '../input/bert-multi/bert_multi_from_tfhub'
RECORDS_GCS_PATH = '../input/tfrecords'

def init():
  # if FLAGS.mode == 'valid' or FLAGS.mode == 'test':
  #   FLAGS.gpus = 1

  if gezi.get_env('VALID_BYLANG') != None:
    FLAGS.valid_bylang = bool(int(gezi.get_env('VALID_BYLANG')))

  if FLAGS.valid_en:
    if FLAGS.test_en is None:
      FLAGS.test_en = True
    FLAGS.valid_input = FLAGS.valid_input.replace('validation', 'validation-en')
    FLAGS.train_input = FLAGS.train_input.replace('validation', 'validation-en')
  if FLAGS.valid_bylang:
    FLAGS.valid_input = FLAGS.valid_input.replace('validation', 'validation-bylang')
    FLAGS.train_input = FLAGS.train_input.replace('validation', 'validation-bylang')
  if FLAGS.test_en:
    FLAGS.test_input = FLAGS.test_input.replace('test', 'test-en')

  if 'bylang' in FLAGS.valid_input:
    if FLAGS.folds is not None:
      FLAGS.folds = 3
    if not FLAGS.model_dir.endswith('bylang'):
      FLAGS.model_dir = f'{FLAGS.model_dir}-bylang'
  if 'pair' in FLAGS.train_input:
    FLAGS.batch_size = int(FLAGS.batch_size / 2)
  # if 'large' in FLAGS.pretrained:
  #   FLAGS.batch_size = int(FLAGS.batch_size / 2)
