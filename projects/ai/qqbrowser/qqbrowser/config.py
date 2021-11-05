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
from omegaconf import OmegaConf
import numpy as np

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import gezi
from gezi import logging
import melt as mt

hugs = {
  'base': 'bert-base-chinese',
  'macbert': 'hfl/chinese-macbert-base',
  'wwm': 'hfl/chinese-bert-wwm-ext',
  'roberta': 'hfl/chinese-roberta-wwm-ext',  #default
  'tiny': 'ckiplab/albert-tiny-chinese',
  'large': 'hfl/chinese-roberta-wwm-ext-large'
}

flags.DEFINE_string('transformer', 'hfl/chinese-macbert-base', '')  
flags.DEFINE_string('hug', None, '')
flags.DEFINE_string('bert_dir', '../input/chinese_L-12_H-768_A-12', '')
flags.DEFINE_bool('from_pt', False, '')
flags.DEFINE_bool('continue_pretrain', False, '')
flags.DEFINE_string('continue_version', None, '')
flags.DEFINE_string('bert_pooling', 'cls', 'cls pooler max mean, att?')
flags.DEFINE_bool('bert_trainable', True, '')
flags.DEFINE_bool('words_bert_trainable', True, '')

flags.DEFINE_integer('max_frames', 32, '')
flags.DEFINE_integer('frame_embedding_size', 1536, '')
flags.DEFINE_integer('max_len', 0, '')
flags.DEFINE_integer('max_title_len', 40, '40 66')
flags.DEFINE_integer('last_title_tokens', 10, '')
flags.DEFINE_integer('max_asr_len', 64, '')
# tfrecords3 --rv3 采用了混合title asr的input方式 总长度96 title不超过32 30 + CLS + SEP， 也就是tfrecords3是有input_ids而tfrecords是title_ids, asr_ids
flags.DEFINE_integer('max_text_len', 96, '')
flags.DEFINE_integer('max_words', 64, '')
flags.DEFINE_integer('asr_len', 0, '')
flags.DEFINE_integer('last_asr_tokens', 20, '')
flags.DEFINE_integer('max_tags', 29, '12 as 99%')
flags.DEFINE_integer('loss_tags', 0, '0 means all, max_tags actually')
flags.DEFINE_bool('hard_tag', False, '')
flags.DEFINE_integer('max_eval_examples', 1000, '')

flags.DEFINE_bool('tag_w2v', False, '')
flags.DEFINE_bool('tag_norm', True, '')
flags.DEFINE_bool('tag_trainable', True, '')

flags.DEFINE_bool('word_w2v', False, '')
flags.DEFINE_bool('word_norm', True, '')
flags.DEFINE_bool('word_trainable', True, '')
flags.DEFINE_integer('word_emb_size', 256, '')
flags.DEFINE_alias('wes', 'word_emb_size')
flags.DEFINE_string('word_pooling', 'att', '')

flags.DEFINE_float('normalloss_rate', 1., '')
flags.DEFINE_integer('num_negs', 5, '')
flags.DEFINE_integer('num_vids', 5, '')
flags.DEFINE_float('contrasive_rate', 0., '')
flags.DEFINE_float('catloss_rate', 0., '')
flags.DEFINE_float('subcatloss_rate', 0., '')
flags.DEFINE_float('zeroloss_rate', 0., '')
flags.DEFINE_float('auxloss_rate', 0., '')
flags.DEFINE_float('self_contrasive_rate', 0., '')
flags.DEFINE_float('mlmloss_rate', 0., '')

flags.DEFINE_integer('vlad_cluster_size', 64, '')
flags.DEFINE_integer('vlad_groups', 8, '')
flags.DEFINE_integer('vlad_expansion', 2, '')
flags.DEFINE_integer('vlad_hidden_size', 1024, '')
flags.DEFINE_integer('vlad_hidden_size2', 1024, '')
flags.DEFINE_integer('se_ratio', 8, '')

flags.DEFINE_integer('final_size', 256, '')
flags.DEFINE_integer('bert_size', 768, '')
flags.DEFINE_integer('hidden_size', 256, '')
flags.DEFINE_float('vlad_dropout', 0., '')
flags.DEFINE_alias('nextvlad_dropout', 'vlad_dropout')
flags.DEFINE_bool('share_vlad', True, '')
flags.DEFINE_alias('share_nextvlad', 'share_vlad')
flags.DEFINE_float('fusion_dropout', 0., '')

flags.DEFINE_bool('use_title', True, '')
flags.DEFINE_bool('use_frames', True, '')
flags.DEFINE_alias('use_vision', 'use_frames')
flags.DEFINE_bool('use_merge', False, '')
flags.DEFINE_integer('merge_method', 0, '0 title + vision, 1 words + vision, 2 title + words + vision')
flags.DEFINE_bool('use_asr', False, '')
flags.DEFINE_bool('use_words', False, '')
flags.DEFINE_bool('merge_vision', False, '')
flags.DEFINE_bool('merge_vision_after_vlad', False, '')
flags.DEFINE_bool('title_vlad', False, '')
flags.DEFINE_alias('title_nextvlad', 'title_vlad')

flags.DEFINE_bool('words_rnn', False, '')
flags.DEFINE_string('words_bert', '', 'tiny or base')
flags.DEFINE_alias('wbert', 'words_bert')

flags.DEFINE_bool('stop_vision', False, '')
flags.DEFINE_bool('stop_text', False, '')

flags.DEFINE_bool('use_first_vision', False, '')
flags.DEFINE_bool('use_vision_encoder', False, '')
flags.DEFINE_integer('vision_layers', 4, '')
flags.DEFINE_float('vision_drop', 0., '')
flags.DEFINE_string('vision_encoder', 'transformer', '')
flags.DEFINE_string('rnn_method', 'bi', '')
flags.DEFINE_string('rnn', 'LSTM', '')
flags.DEFINE_bool('lm_target', False, '')

flags.DEFINE_float('bert_lr', 3e-5, '')
flags.DEFINE_bool('use_bert_lr', False, '')

flags.DEFINE_bool('use_temperature', False, '')

flags.DEFINE_string('label_strategy', 'selected_tags', '')
flags.DEFINE_integer('num_labels', 10000, '')
flags.DEFINE_string('multi_label_file', '../input/tag_list.txt', '')

flags.DEFINE_bool('l2_norm', False, '')
flags.DEFINE_bool('contrasive_norm', True, '')
flags.DEFINE_bool('from_logits', False, '')
flags.DEFINE_string('loss_fn', 'binary', '')
flags.DEFINE_string('tag_pooling', None, '')
flags.DEFINE_string('tag_pooling2', 'mean', 'for contrasive learning')
flags.DEFINE_bool('arc_face', False, '')
flags.DEFINE_float('arcface_rate', 0., '')

flags.DEFINE_bool('loss_sum_byclass', True, '')
flags.DEFINE_string('final_activation', 'sigmoid', '')
flags.DEFINE_alias('last_activation', 'final_activation')
flags.DEFINE_bool('mdrop', False, '')
flags.DEFINE_bool('mdrop2', False, '')
flags.DEFINE_float('mdrop_rate', 0., '')
flags.DEFINE_integer('mdrop_experts', 5, '')

flags.DEFINE_string('pooling', None, '')

flags.DEFINE_bool('baseline_dataset', False, '')

flags.DEFINE_bool('dump_records', False, '')
flags.DEFINE_bool('loop_records', False, '')

flags.DEFINE_integer('parse_strategy', 1, '')
flags.DEFINE_string('records_name', 'tfrecords', '')
flags.DEFINE_string('records_version', '0', '')
flags.DEFINE_alias('rv', 'records_version')
flags.DEFINE_bool('use_pyfunc', False, '')
flags.DEFINE_bool('remove_pred', False, '')

flags.DEFINE_string('activation', 'relu', '')
flags.DEFINE_bool('activate_last', True, '')
flags.DEFINE_bool('activate_last2', False, '')

flags.DEFINE_integer('fold_', 0, '0 or 1')
flags.DEFINE_integer('folds_', 5, '3 only.., valid, train, train_all')

flags.DEFINE_bool('log_image', False, '')
flags.DEFINE_integer('vis_seed', 521, '')
flags.DEFINE_integer('num_random', 20, '')
flags.DEFINE_integer('num_worst', 40, '')
flags.DEFINE_integer('num_best', 10, '')

flags.DEFINE_bool('tag_softmax', False, '')
flags.DEFINE_integer('top_tags', 0, '')

flags.DEFINE_float('from_logits_mask', 0., '')

flags.DEFINE_list('mlp_dims', [], '')
flags.DEFINE_list('mlp_dims2', [], '')

flags.DEFINE_bool('perm', False, '')

flags.DEFINE_bool('online', False, '')
flags.DEFINE_bool('use_se', True, '')
flags.DEFINE_bool('sfu', False, '')
flags.DEFINE_bool('cosine_loss', False, '')

flags.DEFINE_bool('numerical_stable', False, '')
flags.DEFINE_bool('vision_dense', False, '')
flags.DEFINE_bool('title_dense', False, '')
flags.DEFINE_bool('batch_norm', False, '')
flags.DEFINE_bool('layer_norm', False, '')
flags.DEFINE_bool('batch_norm2', False, '')
flags.DEFINE_bool('layer_norm2', False, '')
flags.DEFINE_bool('final_dense', False, '')
flags.DEFINE_bool('gem_norm', False, '')

flags.DEFINE_bool('finetune', False, '')
flags.DEFINE_alias('ft', 'finetune')

flags.DEFINE_float('ft_epochs', 5., '')
flags.DEFINE_float('ft_decay_epochs', 0, '')
flags.DEFINE_string('ft_loss_fn', 'corr', '')
flags.DEFINE_float('ft_loss_scale', 1., '')
flags.DEFINE_float('ft_lr_mul', 1., '')
flags.DEFINE_string('ft_mns', '.ft', '')
flags.DEFINE_integer('ft_nvs', 5, '')
flags.DEFINE_float('ft_bs_mul', 1., '')
flags.DEFINE_integer('online_nvs', 2, '')
flags.DEFINE_bool('ft_log_image', True, '')
flags.DEFINE_float('ft_min_lr', None, '')
flags.DEFINE_bool('one_tower', False, '')
flags.DEFINE_string('one_tower_pooling', 'concat,dot2', '')
flags.DEFINE_bool('one_tower_cls', False, '')
flags.DEFINE_integer('num_relevances', 21, '0, 0.05, ... 0.95, 1')

flags.DEFINE_integer('num_clusters', None, '')

flags.DEFINE_float('pos_weight', 0.8, '')
flags.DEFINE_bool('pairwise_ext', False, '')
flags.DEFINE_alias('ft_ext', 'pairwise_ext')
flags.DEFINE_string('ft_ext_mark', 'new', 'new new2 or new,new2 means no 0 an 1')

flags.DEFINE_bool('mask_words', False, '')
flags.DEFINE_bool('merge_text', False, 'True for tfrecords using iput_ids instead of title_ids asr_dis')

flags.DEFINE_bool('weight_loss', False, '')
flags.DEFINE_integer('weight_method', 0, '')
flags.DEFINE_float('weight_power', 1., '')
flags.DEFINE_bool('adjust_label', False, '')
flags.DEFINE_bool('use_relevance2', False, '')
flags.DEFINE_string('relevance', 'relevance', '')

flags.DEFINE_bool('ext_weight', True, '')

flags.DEFINE_float('temperature', 1., '')
flags.DEFINE_float('dynamic_temperature', None, '')

flags.DEFINE_integer('pred_adjust', 0, '0 means do nothing, means max(pred, 0)')

flags.DEFINE_string('segmentor', 'jieba', 'jieba or sp')
flags.DEFINE_bool('mix_segment', False, '')
flags.DEFINE_integer('num_segmentors', 1, '')

flags.DEFINE_bool('swap_train_valid', False, '')
flags.DEFINE_bool('title_lm_only', True, '')
flags.DEFINE_integer('vocab_size', 100000, '')
flags.DEFINE_integer('reserve_vocab_size', 200, '')

# got by caculating each label ratio, see jupter/pairwise-label.ipynb
THRES = [0.28652483836286247,
 0.4145421876610848,
 0.45501038306897,
 0.48457635605826294,
 0.5057033240548461,
 0.5224524661629774,
 0.5404497857111298,
 0.56200385867244,
 0.5844268693206085,
 0.6176600539035921,
 0.6646563277809688,
 0.7041230356853563,
 0.7337663294010222,
 0.7600222389136807,
 0.7788259031797229,
 0.7922097527209533,
 0.8062710791027852,
 0.8238928408371258,
 0.8427406883753812,
 0.891180282478387,
 0.9652019911928011]

def init():
  np.random.seed(1024)
  FLAGS.records_name += FLAGS.records_version
  ic(FLAGS.records_name)
  FLAGS.static_input = True
  # if FLAGS.parse_strategy <= 2:
  FLAGS.batch_parse = False # 因为有parse_label 目前至少是需要tf.py_function所以不能batch_parse

  if mt.get_mode() == 'train':
    FLAGS.do_test = False

  if mt.get_mode() == 'test':
    FLAGS.remove_pred = True

  if FLAGS.lm_target:
    FLAGS.use_bert_lr = False

  if FLAGS.hug:
    FLAGS.transformer = hugs[FLAGS.hug] 
    FLAGS.mn += f'.{FLAGS.hug}'

  if FLAGS.words_bert:
    if FLAGS.words_bert == 'base':
      FLAGS.transformer = 'bert-base-chinese'
    else:
      FLAGS.transformer = 'hfl/chinese-roberta-wwm-ext-large'

  if 'large' in FLAGS.transformer:
    FLAGS.batch_size = int(FLAGS.batch_size / 2)

  if int(FLAGS.rv) >= 4:
    FLAGS.segmentor = 'sp'

  if FLAGS.mn:
    if FLAGS.rv and not '.rv' in FLAGS.mn:
      FLAGS.mn += f'.rv{FLAGS.rv}'

    if FLAGS.incl:
      incls= ','.join(FLAGS.incl)
      FLAGS.mn += f'.incl_{incls}'
    
    if FLAGS.excl:
      excls= ','.join(FLAGS.excl)
      FLAGS.mn += f'.excl_{excls}'

  if FLAGS.use_words:
    FLAGS.mn += f'.{FLAGS.wes}'
  
  if FLAGS.wbert:
    FLAGS.mn += f'.wbert-{FLAGS.wbert}'

  if FLAGS.merge_method > 0:
    FLAGS.mn += f'.m{FLAGS.merge_method}'

  if not FLAGS.title_lm_only:
    FLAGS.mn += '.ntlo'

  if len(FLAGS.incl) == 1:
    incl = FLAGS.incl[0]
    if 'vision' in incl:
      FLAGS.use_bert_lr = False
    if 'word' in incl and (not 'bert' in incl):
      FLAGS.use_bert_lr = False

  ic(FLAGS.incl, FLAGS.use_bert_lr)

  if FLAGS.num_gpus !=4 and int(FLAGS.rv) % 2 == 1:
    if not FLAGS.max_len:
      FLAGS.max_len = 64
    if FLAGS.word_emb_size >= 512:
      FLAGS.batch_size = int(FLAGS.batch_size / 2)

  if FLAGS.finetune:
    FLAGS.mlmloss_rate = 0.
    FLAGS.first_interval_epoch = 0.1
    FLAGS.parse_strategy = 3
    FLAGS.reset_all = True
    FLAGS.ev_first = True
    if FLAGS.ft_lr_mul != 1:
      FLAGS.lr *= FLAGS.ft_lr_mul 
      FLAGS.bert_lr *= FLAGS.ft_lr_mul
    if FLAGS.ft_min_lr is not None:
      FLAGS.min_learning_rate = FLAGS.ft_in_lr
    FLAGS.loss_scale = FLAGS.ft_loss_scale
    FLAGS.from_logits = False
    FLAGS.decay_epochs = FLAGS.ft_decay_epochs
    FLAGS.epochs = FLAGS.ft_epochs
    FLAGS.loss_fn = FLAGS.ft_loss_fn
    FLAGS.nvs = FLAGS.ft_nvs # num valid steps
    FLAGS.log_image = FLAGS.ft_log_image
    FLAGS.bs_mul = FLAGS.ft_bs_mul
    if FLAGS.fold_ != 0:
      FLAGS.log_image = False
    FLAGS.contrasive_rate = 0.
    FLAGS.self_contrasive_rate = 0.
    FLAGS.fusion_dropout = 0.
    FLAGS.vlad_dropout = 0.
    FLAGS.mdrop_rate = 0.

    FLAGS.batch_size = int(FLAGS.batch_size / 2)
  else:
    if mt.get_mode() != 'train':
      FLAGS.mlmloss_rate = 0.

    if FLAGS.mlmloss_rate > 0.:
      FLAGS.batch_size = int(FLAGS.batch_size / 2)
      FLAGS.drop_remainder = True # HACK for get_masked_lm_fn need fixed batch_size for scatter_nd TODO

  ic(FLAGS.num_gpus, FLAGS.rv, FLAGS.max_len, FLAGS.batch_size)

  if FLAGS.parse_strategy == 1:
    # tagid 到 selected_tag 转换变成了固定长度 不再需要padded batch
    # 另外padded batch会极大降低速度 还是固定长度比较好
    if os.path.exists('../input/num_train.txt') and mt.get_mode() != 'test':
      FLAGS.num_train = gezi.read_int('../input/num_train.txt')
      FLAGS.num_valid = gezi.read_int('../input/num_valid.txt')
      # FLAGS.num_test = gezi.read_int('../input/num_test.txt')
    else:
      FLAGS.recount_tfrecords = True
  else:
    FLAGS.label_strategy = 'all_tags'
    if FLAGS.train_input:
      FLAGS.train_input = FLAGS.train_input.replace('tfrecords', FLAGS.records_name)
      FLAGS.valid_input = FLAGS.valid_input.replace('tfrecords', FLAGS.records_name)
      FLAGS.test_input = FLAGS.test_input.replace('tfrecords', FLAGS.records_name)
      
  if FLAGS.use_bert_lr:
    FLAGS.learning_rates = [FLAGS.lr, FLAGS.bert_lr]
    
  # pairwise    
  if FLAGS.parse_strategy == 3:
    files = gezi.list_files(f'../input/{FLAGS.records_name}/pairwise/*.tfrec')
    fold = FLAGS.fold_ 
    FLAGS.valid_files = gezi.list_files(f'../input/{FLAGS.records_name}/pairwise/valid{fold}/*.tfrec')
    FLAGS.train_files = gezi.list_files(f'../input/{FLAGS.records_name}/pairwise/train{fold}/*.tfrec')
    if FLAGS.swap_train_valid:
      FLAGS.train_files, FLAGS.valid_files = FLAGS.valid_files, FLAGS.train_files
    if FLAGS.pairwise_ext:
      FLAGS.train_files += gezi.list_files(f'../input/{FLAGS.records_name}/pairwise/{FLAGS.ft_ext_mark}_train{fold}/*.tfrec')
    ic(FLAGS.valid_files[:2], FLAGS.train_files[:2], FLAGS.pretrain)
  else:
    # pointwise not log image
    FLAGS.log_image = False
    
  if FLAGS.online:
    FLAGS.wandb = False
    FLAGS.log_image = False
    FLAGS.allow_valid_train = True
    if FLAGS.parse_strategy == 3:
      # FLAGS.train_files = files
      ## 2 means all pairwise instances
      FLAGS.train_files = gezi.list_files(f'../input/{FLAGS.records_name}/pairwise/all/*.tfrec')
      if FLAGS.pairwise_ext:
        FLAGS.train_files += gezi.list_files(f'../input/{FLAGS.records_name}/pairwise/{FLAGS.ft_ext_mark}/*.tfrec')

    if FLAGS.vie <= 1:
      FLAGS.vie = 20
    FLAGS.model_dir = FLAGS.model_dir.replace('offline', 'online')
    if FLAGS.pretrain:
      FLAGS.pretrain = FLAGS.pretrain.replace('offline', 'online')

    FLAGS.num_valid_steps = FLAGS.online_nvs

  if FLAGS.mlp_dims:
    FLAGS.mlp_dims = [int(x) for x in FLAGS.mlp_dims]
  if FLAGS.mlp_dims2:
    FLAGS.mlp_dims2 = [int(x) for x in FLAGS.mlp_dims2]

  if FLAGS.hidden_size != FLAGS.final_size:
    FLAGS.final_dense = True

  if FLAGS.arc_face:
    FLAGS.l2_norm = True
    if not FLAGS.ft:
      FLAGS.loss_fn = 'multi'
    FLAGS.from_logits = True
    FLAGS.use_temperature = False

  if FLAGS.cosine_loss:
    FLAGS.l2_norm = True
    FLAGS.from_logits = True

  # for compat old scripts
  if FLAGS.relevance == 'relevance':
    if FLAGS.use_relevance2:
      FLAGS.relevance = 'relevance2'
    if FLAGS.adjust_label:
      FLAGS.relevance = 'label'

  if FLAGS.words_bert:
    FLAGS.use_words = False

  ic(FLAGS.transformer, FLAGS.relevance, FLAGS.use_words, 
     FLAGS.words_bert, FLAGS.word_emb_size,
     FLAGS.parse_strategy, FLAGS.label_strategy, 
     FLAGS.use_pyfunc, FLAGS.remove_pred, 
     FLAGS.from_logits, FLAGS.last_activation, 
     FLAGS.loss_fn, FLAGS.loss_scale,
     FLAGS.batch_parse, FLAGS.l2_norm,
     FLAGS.tag_w2v, FLAGS.tag_norm, FLAGS.tag_trainable,
     FLAGS.word_w2v, FLAGS.word_norm, FLAGS.word_trainable,
     FLAGS.fold_, FLAGS.log_image)
  
