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

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('loss', 'loss_fn', '')
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

flags.DEFINE_bool('dynamc_watch', False, '')

flags.DEFINE_string('lm_target', None, '')

flags.DEFINE_integer('vocab_size', 1100000, 'min count 5 110w, min count 10 60w, total cid 3412774, total ad id 3027362')

flags.DEFINE_bool('use_mask', True, '')

# for vid
flags.DEFINE_bool('use_w2v', False, '')
flags.DEFINE_string('vid_pretrain', '../input/all/glove-128/emb.npy', '')
flags.DEFINE_bool('train_vid_emb', True, '')
# for words
flags.DEFINE_bool('words_w2v', False, '')
flags.DEFINE_string('words_pretrain', '../input/all/glove-words-128/emb.npy', '')
flags.DEFINE_bool('train_word_emb', True, '')

flags.DEFINE_bool('stars_w2v', False, '')
flags.DEFINE_string('stars_pretrain', '../input/all/glove-stars-128/emb.npy', '')
flags.DEFINE_bool('train_stars_emb', True, '')

flags.DEFINE_bool('train_image_emb', False, '')

flags.DEFINE_integer('max_vid', 0, '')
flags.DEFINE_integer('max_words', 0, '')

flags.DEFINE_integer('day', 0, '')
flags.DEFINE_integer('end_', 29, '')
flags.DEFINE_integer('span', 29, 'meaning all using 30')
flags.DEFINE_bool('mix_train', False, '')

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

flags.DEFINE_float('label_smoothing_rate', 0., '')
flags.DEFINE_float('unk_aug_rate', 0., '')

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

flags.DEFINE_bool('use_uid', False, '')
flags.DEFINE_bool('use_vid', False, '')
flags.DEFINE_bool('use_uinfo', False, '')
flags.DEFINE_bool('use_vinfo', False, '')
flags.DEFINE_bool('use_history', False, '')

flags.DEFINE_bool('use_dense_common', False, '')
flags.DEFINE_bool('use_dense_history', False, '')
flags.DEFINE_bool('use_dense_his_durs', False, '')
flags.DEFINE_bool('use_dense_his_freshes', False, '')
flags.DEFINE_bool('use_dense_prev_info', False, '')
flags.DEFINE_bool('use_dense_his_clicks', False, '')
flags.DEFINE_bool('use_dense_match', False, '')
flags.DEFINE_bool('use_dense_his_interval', False, '')

flags.DEFINE_bool('use_last_vid', False, '')
flags.DEFINE_bool('use_prev_info', False, '')
flags.DEFINE_bool('use_class_info', False, '')
flags.DEFINE_bool('use_history_class', False, '')
flags.DEFINE_bool('use_last_class', False, '')
flags.DEFINE_bool('use_story', False, '')
flags.DEFINE_bool('use_title', False, '')
flags.DEFINE_bool('use_stars', False, '')
flags.DEFINE_bool('use_first_star', False, '')
flags.DEFINE_bool('use_image', False, '')
flags.DEFINE_bool('use_others', False, '')
flags.DEFINE_bool('use_dense', False, '')

flags.DEFINE_bool('use_his_durs', False, '')
flags.DEFINE_bool('use_his_freshes', False, '')
flags.DEFINE_bool('use_match_cats', False, '')
flags.DEFINE_bool('use_his_clicks', False, '')
flags.DEFINE_bool('use_his_cats', True, '')
flags.DEFINE_bool('use_his_image', False, '')
flags.DEFINE_bool('use_titles', False, '')
flags.DEFINE_bool('use_stars_list', False, '')
flags.DEFINE_bool('use_match_stars', False, '')
flags.DEFINE_bool('use_latest_stars', False, '')
flags.DEFINE_bool('use_all_stars', False, '')
flags.DEFINE_bool('use_last_feats', True, '')
flags.DEFINE_bool('use_match', False, '')
flags.DEFINE_bool('use_active', True, '')
flags.DEFINE_bool('use_his_intervals', False, '')

flags.DEFINE_bool('use_last_stars', False, '')
flags.DEFINE_bool('use_last_title', False, '')

flags.DEFINE_bool('use_position_emb', False, '')
flags.DEFINE_bool('use_time_emb', False, '')


flags.DEFINE_bool('no_wv_len', False, '')

flags.DEFINE_bool('use_unk', False, '')

context_cols = ['prev', 'mod', 'mf', 'aver', 'sver', 'region']
item_cols = ['vid', 'duration_', 'title_length_', 'cid', 'class_id', 'second_class', 'is_intact', 'vv_', 'ctr_']
ignored_cols = set(['vid', 'cid', 'class_id', 'second_class'])

def init():
  if FLAGS.toy:
    FLAGS.train_input = FLAGS.train_input.replace('tfrecords', 'tfrecords-toy')
    FLAGS.min_tfrecords=10

  if FLAGS.lm_target:
    FLAGS.day = 0
    FLAGS.train_hour = '0'
  
    dirs = [f'../input/tfrecords-lm/train/{i + 1}' for i in range(30)]
    dirs = ['../input/tfrecords-lm/eval'] + dirs
    if not FLAGS.lm_all_day:
      FLAGS.train_input = ','.join(dirs[:-1])
      FLAGS.valid_input = dirs[-1]
    else:
      FLAGS.train_input = ','.join(dirs)
      FLAGS.valid_input = None

  if FLAGS.day:
    FLAGS.loop_train = False
    valid_day = min(FLAGS.day + FLAGS.span, 30)
    FLAGS.valid_input = f'{FLAGS.train_input}/{valid_day}'
    FLAGS.train_input = f'{FLAGS.train_input}/{FLAGS.day}'
    
    if FLAGS.day != 30:
      FLAGS.test_input = None
    else:
      FLAGS.valid_input = None

  # mixed train start from FLAGS.mix_train
  if FLAGS.mix_train:
    FLAGS.loop_train = False
    FLAGS.valid_input = f'{FLAGS.train_input}/30'
    # not to use 29 for train as eval/test dataset actually is day 32
    FLAGS.train_input = ','.join([f'{FLAGS.train_input}/{i}' for i in range(int(FLAGS.start), FLAGS.end_)])
    FLAGS.mname += '.mix'

  # 自动循环train/valid/test 注意始终使用day=30做验证
  day = int(FLAGS.start)
  valid_day = 30
  FLAGS.mname += f'.{day}'
  if FLAGS.mode != 'train':
    FLAGS.valid_hour = '30'

  # 必须有这个。。
  if FLAGS.loop_range:
    FLAGS.valid_span = valid_day - day

  try:
    train_uids = pickle.load(open(f'../input/train/part_{valid_day - 1}/dids.pkl', 'rb'))
    valid_uids = pickle.load(open(f'../input/train/part_{valid_day}/dids_1.pkl', 'rb'))

    new_uids = set(valid_uids).difference(set(train_uids))
    # print('new_uids ratio', len(new_uids) / len(valid_uids), len(new_uids), len(valid_uids))
    gezi.set('new_uids', new_uids)

    train_vids = pickle.load(open(f'../input/train/part_{valid_day - 1}/vids.pkl', 'rb'))
    valid_vids = pickle.load(open(f'../input/train/part_{valid_day}/vids_1.pkl', 'rb'))

    new_vids = set(valid_vids).difference(set(train_vids))
    # print('new_vids ratio', len(new_vids) / len(valid_vids), len(new_vids), len(valid_vids))
    gezi.set('new_vids', new_vids)
  except Exception:
    pass

  vocab_names = [
                'did', 'vid', 'words', 'stars', 'did', 'region', 'sver', 
                'mod', 'mf', 'aver', 'is_intact', 'second_class', 'class_id', 'cid',
              ]
  vocab_sizes = {}
  for vocab_name in vocab_names:
    vocab = gezi.Vocab(f'../input/all/{vocab_name}.txt')
    min_count = FLAGS.min_count
    vocab_size = [vocab.size(), vocab.size(min_count)]
    if vocab_name == 'vid' and FLAGS.max_vid:
      vocab_size[1] = FLAGS.max_vid 
    vocab_sizes[vocab_name] = vocab_size
    # print(vocab_name, vocab_size)
  vocab_sizes['image'] = vocab_sizes['vid']
  gezi.set('vocab_sizes', vocab_sizes)

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

