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

import pickle
from tqdm import tqdm

import gezi
logging = gezi.logging

from projects.ai.mind.prepare.config import *

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_string('loss_type', 'sigmoid', '')
flags.DEFINE_integer('emb_size', 128, '')
flags.DEFINE_integer('hidden_size', 128, '')
flags.DEFINE_string('pooling', 'sum', '')
flags.DEFINE_string('feat_pooling', 'dot', '')
flags.DEFINE_string('encoder', 'GRU', '')
flags.DEFINE_integer('max_len', 5000, '')
flags.DEFINE_float('dropout', 0., '')
flags.DEFINE_float('rdropout', 0., '')
flags.DEFINE_float('mlp_dropout', 0., '')

flags.DEFINE_integer('num_layers', 1, '')
flags.DEFINE_bool('concat_layers', True, '')

flags.DEFINE_bool('self_match', False, '')

flags.DEFINE_integer('num_heads', 4, '')
flags.DEFINE_float('transformer_dropout', 0., '')
flags.DEFINE_bool('transformer_att', None, '')

flags.DEFINE_bool('din_normalize', False, '')

flags.DEFINE_bool('l2_normalize_before_pooling', False, '')

flags.DEFINE_string('lm_target', None, '')

flags.DEFINE_bool('use_mask', True, '')

flags.DEFINE_bool('use_entities', True, '')

flags.DEFINE_bool('use_entity_pretrain', False, '')
flags.DEFINE_string('entity_pretrain', '../input/entity_emb.npy', '')
flags.DEFINE_bool('train_entity_emb', True, '')

flags.DEFINE_bool('use_word_pretrain', False, '')
flags.DEFINE_string('word_pretrain', '../input/glove-100/emb.npy', '')
flags.DEFINE_bool('train_word_emb', True, '')

flags.DEFINE_bool('use_did_pretrain', False, '')
flags.DEFINE_string('did_pretrain', '../input/did-glove-100/emb.npy', '')
flags.DEFINE_bool('train_did_emb', True, '')

flags.DEFINE_string('title_lookup', '../input/title_lookup.npy', '')
flags.DEFINE_string('doc_lookup', '../input/doc_lookup.npy', '')

flags.DEFINE_integer('max_vid', 0, '')
flags.DEFINE_integer('max_words', 0, '')
## in prepare/config.py
# flags.DEFINE_integer('max_history', 0, '')
# flags.DEFINE_integer('max_titles', 0, '')

flags.DEFINE_integer('day', 0, '')
flags.DEFINE_integer('span', 5, 'meaning all using 6')

flags.DEFINE_bool('toy', False, '')
flags.DEFINE_integer('min_count', 0, '')

flags.DEFINE_bool('batch_norm', False, '')

flags.DEFINE_string('activation', 'relu', '')
flags.DEFINE_string('att_activation', 'sigmoid', 'leaky better, or prelu, dice output result nan TODO')
flags.DEFINE_string('transformer_activation', None, 'lekay')

flags.DEFINE_string('his_encoder', None, '')
flags.DEFINE_integer('his_pooling_heads', 4, '')
flags.DEFINE_string('his_pooling', '', '')
flags.DEFINE_string('his_pooling2', '', '')
flags.DEFINE_string('his_simple_pooling', 'sum', '')
flags.DEFINE_string('title_encoder', None, '')
flags.DEFINE_string('title_pooling', 'att', '')
flags.DEFINE_bool('title_att', False, '')
flags.DEFINE_string('story_encoder', None, '')
flags.DEFINE_string('image_pooling', 'mean', '')
flags.DEFINE_string('image_encoder', None, '')
flags.DEFINE_string('stars_encoder', None, '')
flags.DEFINE_string('stars_pooling', None, '')
flags.DEFINE_string('stars_att_pooling', None, '')
flags.DEFINE_bool('stars_att', False, '')

flags.DEFINE_bool('use_weight', False, '')
flags.DEFINE_float('weight_power', 1.0, '')

# flags.DEFINE_bool('use_cur_emb', False, '')

flags.DEFINE_bool('use_vocab_emb', False, '')

flags.DEFINE_string('his_strategy', '1', '')

flags.DEFINE_float('aux_loss_rate', 0., '')
flags.DEFINE_bool('use_bias_emb', False, '')
flags.DEFINE_bool('use_scale_emb', False, '')


flags.DEFINE_bool('big_mlp', True, '')

flags.DEFINE_bool('big_model', False, '')

flags.DEFINE_bool('uv_only', False, '')

flags.DEFINE_bool('use_contexts', False, '')
flags.DEFINE_bool('use_items', False, '')
flags.DEFINE_bool('use_crosses', False, '')

flags.DEFINE_bool('use_shows', False, '')

flags.DEFINE_bool('use_his_concat', False, '')

flags.DEFINE_bool('lm_all_day', False, '')

flags.DEFINE_integer('cross_height', 5000000, '')
flags.DEFINE_integer('num_buckets', 500000, '')

flags.DEFINE_bool('use_uid', True, '')
flags.DEFINE_bool('use_did', True, '')
flags.DEFINE_bool('use_uinfo', False, '')
flags.DEFINE_bool('use_history', False, '')
flags.DEFINE_bool('use_history_info', False, '')
flags.DEFINE_bool('use_history_entities', True, '')
flags.DEFINE_bool('use_news_info', False, '')
flags.DEFINE_bool('use_dense', False, '')
flags.DEFINE_bool('use_time', False, '')
flags.DEFINE_bool('use_fresh', True, '')
flags.DEFINE_bool('use_position', False, '')
flags.DEFINE_bool('use_title', False, '')
flags.DEFINE_bool('use_impressions', False, '')

flags.DEFINE_bool('use_position_emb', False, '')
flags.DEFINE_bool('use_time_emb', False, '')
flags.DEFINE_bool('use_fresh_emb', False, '')

flags.DEFINE_bool('no_wv_len', False, '')

flags.DEFINE_bool('use_unk', False, '')

flags.DEFINE_bool('small', False, '')

flags.DEFINE_bool('train_mask_dids', True, '')
flags.DEFINE_bool('valid_mask_dids', True, '')
flags.DEFINE_bool('test_mask_dids', True, '')
flags.DEFINE_float('mask_dids_ratio', -1., 'need to do exp # dev new did ratio 0.054, test is 0.872')
flags.DEFINE_float('mask_uids_ratio', 0., '')
flags.DEFINE_bool('test_all_mask', False, 'TODO test later submit with test_all_mask=True')

flags.DEFINE_float('label_smoothing_rate', 0., '')
flags.DEFINE_float('unk_aug_rate', 0., '')
flags.DEFINE_alias('unk_aug_ratio', 'unk_aug_rate')

flags.DEFINE_float('neg_mask_ratio', 0., '')
flags.DEFINE_float('neg_filter_ratio', 0., '')

flags.DEFINE_bool('train_uid_emb', True, '')

flags.DEFINE_bool('use_multi_dropout', False, '')

flags.DEFINE_bool('min_count_unk', True, '')

flags.DEFINE_string('input_dir', '../input', '')

flags.DEFINE_bool('slim_emb_height', False, '')

flags.DEFINE_bool('mask_history', True, '')

doc_feats = ['cat', 'sub_cat', 'title']
doc_feat_lens = [1, 1, 30]
doc_feat_len = sum(doc_feat_lens)

def init():
  vocab_names = [
                  'did', 'uid',
                  'cat', 'sub_cat',
                  'entity', 'entity_type',
                  'word'
                ]

  # vocabs = 
  #   {
  #     'uid': {
  #       'min_count': FLAGS.min_count,
  #       'slim': False,
  #       'trainable': FLAGS.train_uid_emb,
  #       'pretrain': None,
  #     },
  #     'did': {
  #       'min_count': FLAGS.min_count,
  #       'slim': False,
  #       'trainable': FLAGS.train_did_emb,
  #       'pretrain': FLAGS.did_pretrain,
  #     },
  #     'cat': {
  #       'min_count': FLAGS.min_count,
  #       'slim': False,
  #       'trainable': True,
  #       'pretrain': None,
  #     },
  #     'sub_cat': {
  #       'min_count': FLAGS.min_count,
  #       'slim': False,
  #       'trainable': True,
  #       'pretrain': None,
  #     },
  #     'entity': {
  #       'min_count': FLAGS.min_count,
  #       'slim': False,
  #       'trainable': FLAGS.train_entity_emb,
  #       'pretrain': FALGS.entity_pretrain,
  #     },
  #     'entity2': {
  #       'min_count': FLAGS.min_count,
  #       'slim': True,
  #       'trainable': True,
  #       'pretrain': None,
  #     },
  #     'entity_type': {
  #       'min_count': FLAGS.min_count,
  #       'slim': False,
  #       'trainable': True,
  #       'pretrain': None,
  #     },
  #     'word': {
  #       'min_count': 0,
  #       'slim': False,
  #       'trainable': FLAGS.train_word_emb,
  #       'pretrain': FLAGS.word_pretrain,
  #     },   
  #   }

  vocab_sizes = {}
  for vocab_name in vocab_names:
    fixed = False if vocab_name != 'word' else True
    vocab = gezi.Vocab(f'{FLAGS.input_dir}/{vocab_name}.txt', fixed=fixed)
    min_count = FLAGS.min_count if vocab_name != 'word' else 0
    logging.debug('---min_count', min_count)
    # # > 1e6 ?????????train??????dev???????????????
    vocab_size = [vocab.size(), vocab.size(min_count + 1000000)]  

    if vocab_name == 'uid' and FLAGS.max_vid:
      vocab_size[1] = FLAGS.max_vid  # vocab_size[1] is not used
    vocab_sizes[vocab_name] = vocab_size
    # print(vocab_name, vocab_size)
  gezi.set('vocab_sizes', vocab_sizes)

  # mixed train start from FLAGS.mix_train
  valid_day = 6
  if FLAGS.mix_train:
    FLAGS.loop_train = False
    FLAGS.valid_input = f'{FLAGS.train_input}/{valid_day}'
    FLAGS.train_input = ','.join([f'{FLAGS.train_input}/{i}' for i in range(int(FLAGS.start), valid_day)])
    FLAGS.mname += '.mix'

  # ????????????train/valid/test ??????????????????day=6?????????
  day = int(FLAGS.start or 0)
  if day != 0:
    FLAGS.mname += f'.{day}'
  if FLAGS.mode != 'train':
    FLAGS.valid_hour = str(valid_day)

  if 'rand' in FLAGS.input:
    FLAGS.shuffle = True

  if 'pad' in FLAGS.input:
    FLAGS.record_padded = True
    
  if FLAGS.neg_mask_ratio > 0:
    FLAGS.use_weight = True

  if FLAGS.big_model:
    FLAGS.his_encoder = 'gru'
    FLAGS.title_encoder = 'gru'
    FLAGS.title_pooling = 'att'
    # FLAGS.use_contexts = True
    FLAGS.use_his_image = True
    FLAGS.use_image = True
    FLAGS.train_image_emb = True
    # FLAGS.image_encoder = 'gru'
    # FLAGS.use_crosses = True

