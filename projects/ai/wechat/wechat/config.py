#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2021-01-10 17:16:24.668713
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import random
from icecream import ic
from omegaconf import OmegaConf

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import gezi
from gezi import logging
import melt as mt

# 初赛待预测行为列表
ACTION_LIST_V1 = ["read_comment", "like", "click_avatar",  "forward"]

# 复赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
ACTIONS = ACTION_LIST

weights_map = {
          "read_comment": 4.,  # 是否查看评论
          "like": 3.,  # 是否点赞
          "click_avatar": 2.,  # 是否点击头像
          "forward": 1.,  # 是否转发
          "favorite": 1.,  # 是否收藏
          "comment": 1.,  # 是否发表评论
          "follow": 1.,  # 是否关注
          "action": 1.,
          "finish": 1.,
          "stay": 1.
      }

DOC_FEATS = [
  'doc',
  'feed',
  'author',
  'song',
  'singer',
  'video_time',
  'video_time2',
  'manual_keys',
  'machine_keys',
  'manual_tags',
  'machine_tags',
  'desc'
]

MAX_TAGS = 6
MAX_KEYS = 7
DESC_LEN = 64
DESC_CHAR_LEN = 128
WORD_LEN = 128
START_ID = 3
UNK_ID = 1
EMPTY_ID = 2
NAN_ID = -1

eval_other_keys = ['action', 'is_first', 'fresh', 'finish', 'stay', 'num_actions', 'num_poss']
eval_keys = list(dict.fromkeys(['userid', 'feedid'] + ACTION_LIST + [f'num_{action}s' for action in ACTION_LIST] + eval_other_keys))
unk_keys = ['song', 'singer', 'manual_tags', 'machine_tags', 'manual_keys', 'machine_keys', 'desc', 'desc_char', 'ocr', 'asr']
single_keys = ['author', 'song', 'singer']
multi_keys =  ['manual_tags', 'machine_tags', 'machine_tag_probs', 'machine_tags2', 'manual_keys', 'machine_keys', 'desc', 'desc_char', 'ocr', 'asr']
info_keys = single_keys + multi_keys
info_lens = [1] * len(single_keys) + [MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_KEYS, MAX_KEYS, DESC_LEN, DESC_CHAR_LEN, WORD_LEN, WORD_LEN]

vocab_names = [
                'user', 'doc',
                'author', 'singer', 'song',
                'key', 'tag', 'word', 'char'
              ]

vocabs = {}

flags.DEFINE_string('transformer', 'bert-base-chinese', '')  
flags.DEFINE_bool('from_pt', False, '')
flags.DEFINE_string('loss', 'loss_fn', '')
flags.DEFINE_bool('mdrop', False, '')
flags.DEFINE_string('dev_pkl', '../input/dev.pkl', '')
flags.DEFINE_string('dev_json', '../input/dev.lat.json', '')
# flags.DEFINE_string('dev_json', '../input/test_data.json', '')

flags.DEFINE_integer('auc_threads', 20, '')
flags.DEFINE_bool('filter_position', False, '')

flags.DEFINE_bool('online', False, '')
flags.DEFINE_list('feats', [], 'user,doc,device,day')
flags.DEFINE_list('feats2', [], 'multi feats like tags')

flags.DEFINE_list('his_id_feats', [], '')
flags.DEFINE_list('his_actions', [], '')
flags.DEFINE_list('his_actions2', [], '')
flags.DEFINE_list('his_user_actions', [], '')

flags.DEFINE_list('his_feats', [], '')
flags.DEFINE_list('his_feats2', [], '')

flags.DEFINE_list('dense_feats', ['video_display'], '')
flags.DEFINE_list('count_feats', [], '')

flags.DEFINE_bool('weight_loss', False, '')
flags.DEFINE_list('weights', [1] * 100, '')
flags.DEFINE_bool('weight_loss_byday', False, '')
flags.DEFINE_float('weight_power', 1.0, '')

flags.DEFINE_bool('use_dense', False, '')

flags.DEFINE_integer('emb_dim', 128, '')
flags.DEFINE_integer('word_emb_dim', 128, '')
flags.DEFINE_integer('char_emb_dim', None, '')
flags.DEFINE_integer('key_emb_dim', None, '')

flags.DEFINE_string('user_emb', None, '')
flags.DEFINE_bool('user_trainable', True, '')
flags.DEFINE_string('doc_emb', None, '')
flags.DEFINE_bool('doc_trainable', True, '')
flags.DEFINE_string('doc2_emb', None, '')
flags.DEFINE_bool('doc2_trainable', True, '')
flags.DEFINE_string('feed_emb', 'feed_embeddings', '')
flags.DEFINE_bool('feed_trainable', False, '')

flags.DEFINE_string('desc_vec_emb', None, '')
flags.DEFINE_bool('desc_vec_trainable', False, '')
flags.DEFINE_string('ocr_vec_emb', None, '')
flags.DEFINE_bool('ocr_vec_trainable', False, '')
flags.DEFINE_string('asr_vec_emb', None, '')
flags.DEFINE_bool('asr_vec_trainable', False, '')

flags.DEFINE_string('author_emb', None, '')
flags.DEFINE_bool('author_trainable', True, '')
flags.DEFINE_string('singer_emb', None, '')
flags.DEFINE_bool('singer_trainable', True, '')
flags.DEFINE_string('song_emb', None, '')
flags.DEFINE_bool('song_trainable', True, '')

flags.DEFINE_list('mlp_dims', [1024, 512, 256], '')
flags.DEFINE_list('task_mlp_dims', [512, 128], '')
flags.DEFINE_bool('task_mlp', False, '')

flags.DEFINE_bool('machine_weights', True, '')

flags.DEFINE_string('tag_pooling', 'att', '')
flags.DEFINE_string('key_pooling', 'att', '')

flags.DEFINE_list('dense_transform_keys', [], '')

flags.DEFINE_list('id_self_keys', [], '')
flags.DEFINE_list('id_return_keys', [], '')
flags.DEFINE_list('id_self_keys2', [], '')
flags.DEFINE_list('id_return_keys2', [], '')

flags.DEFINE_string('encoder_pooling', 'att', '')
flags.DEFINE_string('his_pooling', 'att', '')
flags.DEFINE_string('his_pooling2', 'att', '')
flags.DEFINE_string('pooling', '', '')
flags.DEFINE_string('pooling2', '', '')
flags.DEFINE_string('seqs_pooling', 'att', '')
flags.DEFINE_string('seqs_pooling2', 'att', '')
flags.DEFINE_string('seqs_encoder', None, '')

flags.DEFINE_list('din_keys', ['feed', 'doc', 'doc2'], '')

flags.DEFINE_bool('share_his_encoder', False, '')
flags.DEFINE_bool('share_tag_encoder', True, '')
flags.DEFINE_bool('share_key_encoder', True, '')
flags.DEFINE_bool('share_text_encoder', False, '')
flags.DEFINE_bool('share_his_pooling', False, '')
flags.DEFINE_bool('share_seqs_pooling', False, '')

flags.DEFINE_string('doc_encoder_pooling', 'att', 'att seems not as good as dense here')
flags.DEFINE_bool('doc_his_encoder', False, '')
flags.DEFINE_bool('share_doc_encoder', True, 'wether all actions share one doc encoder')
flags.DEFINE_bool('share_doc_his_encoder', False, '')

flags.DEFINE_integer('max_his', 0, '50')
flags.DEFINE_integer('max_his2', 0, '20')
flags.DEFINE_integer('max_his3', 10, '')
flags.DEFINE_integer('max_texts', 5, '')

flags.DEFINE_string('records_name', 'tfrecords', '')
flags.DEFINE_string('records_name2', 'tfrecords', '')
flags.DEFINE_integer('neg_parts', 4, '')
flags.DEFINE_integer('neg_part', None, '')

flags.DEFINE_integer('valid_day', 14, '')
flags.DEFINE_integer('test_day', 15, '')

flags.DEFINE_bool('use_doc_emb', True, '')
flags.DEFINE_bool('use_user_emb', True, '')

flags.DEFINE_string('word_emb', None, '')
flags.DEFINE_bool('word_trainable', True, '')

flags.DEFINE_string('char_emb', None, '')
flags.DEFINE_bool('char_trainable', True, '')

flags.DEFINE_string('key_emb', None, '')
flags.DEFINE_bool('key_trainable', True, '')

flags.DEFINE_string('tag_emb', None, '')
flags.DEFINE_bool('tag_trainable', True, '')

flags.DEFINE_integer('his_min', 1, '')

flags.DEFINE_string('his_encoder', None, '')
flags.DEFINE_string('his_encoder2', None, '')

flags.DEFINE_integer('num_heads', 2, '')
flags.DEFINE_integer('num_layers', 2, '')

flags.DEFINE_float('dropout', 0., '')
flags.DEFINE_float('emb_dropout', 0., '')
flags.DEFINE_float('pooling_dropout', 0., '')
flags.DEFINE_bool('concat_layers', True, '')

flags.DEFINE_bool('mask_zero', False, '')

flags.DEFINE_integer('min_count', 5, '')

flags.DEFINE_bool('rare_unk', False, '')
flags.DEFINE_list('mask_keys', [], '')
flags.DEFINE_bool('eval_mask_key', True, '')
flags.DEFINE_float('mask_rate', 0., 'if == 0 mask by min_count else mask by prob of mask_rate')
flags.DEFINE_float('mask_his_rate', 0., '0.01')

flags.DEFINE_bool('use_pr_embedding', False, '')

flags.DEFINE_bool('train_byday', False, '')
flags.DEFINE_alias('byday', 'train_byday')

flags.DEFINE_integer('max_his_days', 10, '')

flags.DEFINE_bool('use_position', False, '')
flags.DEFINE_bool('use_span', False, '')

flags.DEFINE_bool('eval_exclnonfirst', True, '')

flags.DEFINE_integer('latest_actions', 0, '')

flags.DEFINE_string('embs_encoder', None, '')
flags.DEFINE_integer('embs_encoder_position', 1, '')

flags.DEFINE_bool('pooling2_ori_embs', False, 'wether pooling to user original embs')

flags.DEFINE_bool('use_transformer', False, '')
flags.DEFINE_bool('use_mhead', False, '')

flags.DEFINE_bool('pos_weight', False, '')

flags.DEFINE_list('action_list', [], '')
flags.DEFINE_list('other_action_list', [], '')
flags.DEFINE_list('loss_list', [], '')
flags.DEFINE_list('other_loss_list', [], '')

flags.DEFINE_bool('mmoe', False, '')
flags.DEFINE_bool('mmoe_mlp', False, '')
flags.DEFINE_float('label_smoothing', 0., '')

flags.DEFINE_bool('hack_input', False, '')
flags.DEFINE_float('embeddings_regularizer', 0., '')
flags.DEFINE_alias('embs_regularizer', 'embeddings_regularizer')
flags.DEFINE_bool('embs_trainable', None, '')

flags.DEFINE_bool('use_tower', False, '')
flags.DEFINE_string('records_version', None, '')
flags.DEFINE_alias('rv', 'records_version')

flags.DEFINE_integer('emb_multiplier', 1, '')
flags.DEFINE_float('emb_stop_rate', 1., '')

flags.DEFINE_string('embeddings_initializer', 'uniform', '')

flags.DEFINE_bool('tower_train', False, '')
# flags.DEFINE_intger('num_negs', 5, '')
flags.DEFINE_string('pretrain_day', '14.5', 'may set 14.5, 14 too see different results on day 14')
flags.DEFINE_string('pretrain_day_online', '15', '')

flags.DEFINE_integer('nw', 0, 'num workers 1 single, 2 using 2, None or 0 try using all cpu cores nw=cpu_count at most if needed')
flags.DEFINE_bool('eval_ab_users', True, '')
flags.DEFINE_bool('always_eval_all', True, '')
flags.DEFINE_bool('simple_eval', False, '')

flags.DEFINE_bool('hack_loss', False, '')
flags.DEFINE_bool('rdrop_loss', False, '')
flags.DEFINE_string('loss_fn', 'bce', '')
flags.DEFINE_bool('reduce_loss', True, '')

flags.DEFINE_bool('batch_norm', False, '')
flags.DEFINE_bool('layer_norm', False, '')

flags.DEFINE_bool('feed_mlp', False, '')
flags.DEFINE_bool('feed_project', False, '')
flags.DEFINE_bool('concat_feed', False, '')

flags.DEFINE_list('doc_keys', ['feed', 'doc2', 'desc_vec', 'ocr_vec', 'asr_vec'], '')

flags.DEFINE_string('mlp_activation', 'relu', '')
flags.DEFINE_bool('mlp_batchnorm', False, '')
flags.DEFINE_bool('mlp_layernorm', False, '')

flags.DEFINE_bool('add_l2_dense', False, '')
flags.DEFINE_bool('l2_norm', True, '')
flags.DEFINE_bool('dice_activation', False, '')

flags.DEFINE_bool('return_sequences', False, '')
flags.DEFINE_alias('return_seqs', 'return_sequences')
flags.DEFINE_bool('return_sequences_only', False, '')
flags.DEFINE_alias('return_seqs_only', 'return_sequences_only')

flags.DEFINE_bool('encode_query', False, '')
flags.DEFINE_bool('add_query_embs', False, '')
flags.DEFINE_bool('add_action_embs', False, '')

flags.DEFINE_integer('his_days', 0, '')
flags.DEFINE_bool('use_mdrop', True, '')

flags.DEFINE_string('rnn_strategy', 'bi', 'bi, forward, backward')
flags.DEFINE_bool('doc_dynamic_feats', False, '')
flags.DEFINE_alias('ddf', 'doc_dynamic_feats')
flags.DEFINE_integer('dynamic_feats_strategy', 1, '')
flags.DEFINE_alias('dfs', 'dynamic_feats_strategy')

flags.DEFINE_bool('mean_unk', False, 'for eval map unk(all 0 emb) to per batch mean emb')
flags.DEFINE_bool('map_unk', True, 'avoid unk match, we turn currently doc UNK 1 -> 0 and levave history UNK as 1')
flags.DEFINE_integer('cross_layers', 0, '')

flags.DEFINE_bool('aug_test', False, '')
flags.DEFINE_bool('aug_valid', False, '')

flags.DEFINE_bool('is_v1', False, '')

flags.DEFINE_bool('fake_label', False, '')
flags.DEFINE_bool('uncertain_loss', False, '')
flags.DEFINE_string('sample_method', None, '')
flags.DEFINE_integer('num_negs', 4, '')
flags.DEFINE_float('aux_loss_rate', 0.1, '')
flags.DEFINE_bool('action_loss', False, '')

flags.DEFINE_string('conf', 'conf', '')

def init():
  conf = OmegaConf.load(f'../input/conf/{FLAGS.conf}.yaml')
  gezi.set('conf', conf)
  ic(FLAGS.conf, conf.his_len['read_comments'])

  if FLAGS.dice_activation:
    gezi.set('activation', 'dice')

  #since here fp16 is slower 。。。 tf2.3 v100 tione TODO
  if 'tione' in os.environ['PATH']: 
    FLAGS.fp16 = False
    FLAGS.wandb = False
    if FLAGS.work_mode == 'train':
      FLAGS.nw = 1 # for safe
  
    # hack for core at last due to tf release gpu resource and multiprocess v100 tf2.3.0
    if not FLAGS.async_valid:
      FLAGS.async_eval = False
      FLAGS.nw = 1

  if 'NODE_NAME' in os.environ:
    if 'p40' in os.environ['NODE_NAME']:
      FLAGS.fp16 = False

    # # A100 not use async valid, as it fast and can use wandb to show metrics
    if 'azwus8f' in os.environ['NODE_NAME']:
      FLAGS.async_valid = False

  if FLAGS.sample_method:
    if 'log' in FLAGS.sample_method:
      FLAGS.batch_parse = False

  if FLAGS.aug_test or FLAGS.aug_valid:
    FLAGS.max_his = 0 
    FLAGS.max_his2 = 0
    FLAGS.max_his3 = 0
    FLAGS.max_texts = 0
    if FLAGS.aug_test:
      FLAGS.mode = 'test'
      FLAGS.online = True
      FLAGS.test_out_file = 'submission2.csv'

    if FLAGS.aug_valid:
      FLAGS.mode = 'valid'
      FLAGS.online = False
      FLAGS.valid_out_file = 'valid2.csv'
    
  FLAGS.weights = list(map(float, FLAGS.weights))
  FLAGS.mlp_dims = list(map(int, FLAGS.mlp_dims))
  FLAGS.task_mlp_dims = list(map(int, FLAGS.task_mlp_dims))

  FLAGS.word_emb_dim = FLAGS.word_emb_dim or FLAGS.emb_dim
  FLAGS.char_emb_dim = FLAGS.char_emb_dim or FLAGS.emb_dim
  FLAGS.key_emb_dim = FLAGS.key_emb_dim or FLAGS.emb_dim

  if FLAGS.records_version != None:
    FLAGS.records_name = f'tfrecords{FLAGS.records_version}'
    FLAGS.records_name2 = f'tfrecords{FLAGS.records_version}'
    mt.append_model_suffix(f'.rv{FLAGS.records_version}')

  FLAGS.valid_input = f'../input/{FLAGS.records_name2}/valid/*.tfrec' 
  FLAGS.test_input = f'../input/{FLAGS.records_name2}/test/*.tfrec'

  def is_train_file(file_):
    v = os.path.basename(os.path.dirname(file_))
    if v in ['valid', 'test']:
      return False
    
    # perslue labeling 看上去没什么用，可以复赛数据再确认
    if v.startswith('pred'):
      if FLAGS.fake_label:
        if FLAGS.online:
          return v == 'pred15'
        else:
          return v == 'pred14'
      else:
        return False

    if FLAGS.online:
      return int(v) < FLAGS.test_day
    else:
      return int(v) < FLAGS.valid_day

  files = gezi.list_files(f'../input/{FLAGS.records_name}/*/*.tfrec')
  FLAGS.input_files = [x for x in files if is_train_file(x)]
  if not FLAGS.train_byday:
    random.seed(FLAGS.seed)
    random.shuffle(FLAGS.input_files)
  else:
    FLAGS.input_files.sort(key=lambda x: int(os.path.basename(os.path.dirname(x))))
  ic(len(FLAGS.input_files), FLAGS.input_files[:5], FLAGS.input_files[-5:])

  FLAGS.save_interval_epochs = FLAGS.vie

  if not FLAGS.action_list:
    FLAGS.action_list = ACTION_LIST.copy() if not FLAGS.is_v1 else ACTION_LIST_V1.copy()
  if FLAGS.other_action_list:
    FLAGS.action_list += FLAGS.other_action_list

  if FLAGS.action_loss:
    FLAGS.action_list.append('action')

  if not FLAGS.loss_list:
    FLAGS.loss_list = FLAGS.action_list.copy()

  if FLAGS.other_loss_list:
    FLAGS.loss_list += FLAGS.other_loss_list

  FLAGS.action_list = list(dict.fromkeys(FLAGS.action_list))
  FLAGS.loss_list = list(dict.fromkeys(FLAGS.loss_list))
  ic(FLAGS.action_list)
  ic(FLAGS.loss_list)
  ic(FLAGS.weights[:len(FLAGS.loss_list)])

  if FLAGS.train_byday:
    # TODO
    FLAGS.shuffle = False
    assert int(FLAGS.num_epochs) == 1
    if not '.ft' in FLAGS.model_name:
      mt.append_model_suffix('.day')
    FLAGS.bert_style_lr = False

  if FLAGS.online:
    FLAGS.do_test = True
    FLAGS.do_valid = False
    FLAGS.valid_input = None
    FLAGS.allow_valid_train = True
    FLAGS.write_valid_final = False
    FLAGS.model_dir = FLAGS.model_dir.replace('offline', 'online')
    FLAGS.num_eval_steps = 0
    FLAGS.wandb_group = os.path.basename(os.path.dirname(FLAGS.model_dir)) + '.online'
    FLAGS.first_interval_epoch = -1
  else:
    FLAGS.do_test = False

  for vocab_name in vocab_names:
    vocab_file =  f'../input/{vocab_name}_vocab.txt'
    vocab = gezi.Vocab(vocab_file)
    vocabs[vocab_name] = vocab
  for name in FLAGS.doc_keys:
    vocabs[name] = vocabs['doc']

  for name in vocabs:
    try:
      emb_name = getattr(FLAGS, f'{name}_emb')
    except Exception:
      emb_name = None
    ic('pretrian_embs', name, emb_name)

  if FLAGS.num_epochs > 1 and not '.epoch' in FLAGS.model_name:
    if not '.ft' in FLAGS.model_name:
      mt.append_model_suffix(f'.ep{int(FLAGS.num_epochs)}')

  if FLAGS.neg_part is not None:
    if not '.ft' in FLAGS.model_name:
      mt.append_model_suffix(f'-{FLAGS.neg_parts}-{FLAGS.neg_part}')

  if FLAGS.tower_train:
    FLAGS.use_tower = True

  # TODO 统计特征似乎有效 可以考虑更好的归一化处理
  if FLAGS.doc_dynamic_feats:
    FLAGS.count_feats.append('num_shows')
    for action in ACTIONS + ['actions']:
      FLAGS.dense_feats.append(f'{action}_rate')
      FLAGS.count_feats.append(f'num_{action}')
  
    for key in ['finish_rate', 'stay_rate']:
      FLAGS.dense_feats.append(f'{key}_mean')
      FLAGS.count_feats.append(f'total_{key}')
    
    for span in [1, 3, 7]:
      FLAGS.count_feats.append(f'num_shows_{span}')
      for action in ACTIONS + ['actions']:
        FLAGS.dense_feats.append(f'{action}_rate_{span}')
        FLAGS.count_feats.append(f'num_{action}_{span}')
  
      for key in ['finish_rate', 'stay_rate']:
        FLAGS.dense_feats.append(f'{key}_mean_{span}')
        FLAGS.count_feats.append(f'total_{key}_{span}')

    FLAGS.dense_feats = list(dict.fromkeys(FLAGS.dense_feats))
    FLAGS.count_feats = list(dict.fromkeys(FLAGS.count_feats))
