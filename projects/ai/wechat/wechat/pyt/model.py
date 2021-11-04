#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-08-01 23:08:36.979020
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import flags
FLAGS = flags.FLAGS

import tensorflow as tf 
import torch
from torch import nn
from torch.nn import functional as F

import gezi
import lele as le

import numpy as np
from wechat.config import *
from wechat.pyt import util 

class Deep(nn.Module):
  def __init__(self):
    super(Deep, self).__init__()
  
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()
    kwargs = dict(padding_idx=0, sparse=FLAGS.sparse_emb, num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner)
    if FLAGS.use_onehot_emb:
      if not gezi.get_global('embedding_keys'):
        self.emb = HashEmbedding(FLAGS.feature_dict_size, FLAGS.hidden_size, large_emb=FLAGS.large_emb, **kwargs)
      else:
        self.emb = le.layers.EmbeddingBags(FLAGS.feature_dict_size, FLAGS.hidden_size, 
                        Embedding=HashEmbedding, split=FLAGS.split_embedding, large_emb=FLAGS.large_emb, **kwargs)
      # self.emb = nn.EmbeddingBag(FLAGS.feature_dict_size, FLAGS.hidden_size, sparse=FLAGS.sparse_emb, mode='sum')
      self.emb_dim = FLAGS.hidden_size
      embedding_keys = gezi.get_global('embedding_keys')
      self.num_fields = FLAGS.field_dict_size - 1 if not embedding_keys else len(embedding_keys)
      logging.debug('deep num_fields:', self.num_fields)
      dim = self.emb_dim  
      if FLAGS.onehot_fields_pooling and 'concat' in FLAGS.fields_pooling:
        dim = self.num_fields * self.emb_dim
    else:
      dim = 0

    if FLAGS.field_emb:
      self.field_emb = nn.Embedding(FLAGS.field_dict_size, FLAGS.hidden_size, padding_idx=0)
      if 'concat' in FLAGS.fields_pooling:
        dim += FLAGS.hidden_size

    if not FLAGS.onehot_fields_pooling:
      self.pooling = le.layers.Pooling(FLAGS.pooling)  

    num_others = 0
    if FLAGS.use_other_embs:
      if FLAGS.use_user_emb:
        self.user_emb = HashEmbeddingUD(FLAGS.feature_dict_size, FLAGS.other_emb_dim, **kwargs)
        num_others += 1
        if 'concat' in FLAGS.fields_pooling:
          dim += FLAGS.other_emb_dim
      
      if FLAGS.use_doc_emb:
        self.doc_emb = HashEmbeddingUD(FLAGS.feature_dict_size, FLAGS.other_emb_dim, **kwargs)
        num_others += 1
        if 'concat' in FLAGS.fields_pooling:
          dim += FLAGS.other_emb_dim
      
      if FLAGS.use_history_emb:
        self.hpooling = le.layers.Pooling(FLAGS.hpooling)
        num_others += 1
        if 'concat' in FLAGS.fields_pooling:
          dim += FLAGS.other_emb_dim

      if FLAGS.use_time_emb:
        self.time_emb = nn.Embedding(500, FLAGS.other_emb_dim)
        self.weekday_emb = nn.Embedding(10, FLAGS.other_emb_dim)
        num_others += 2
        if 'concat' in FLAGS.fields_pooling:
          dim += 2 * FLAGS.other_emb_dim

      if FLAGS.use_timespan_emb:
        self.timespan_emb = nn.Embedding(300, FLAGS.other_emb_dim)
        num_others += 1
        if 'concat' in FLAGS.fields_pooling:
          dim += FLAGS.other_emb_dim

      if FLAGS.use_product_emb:
        self.product_emb = nn.Embedding(10, FLAGS.other_emb_dim)
        num_others += 1
        if 'concat' in FLAGS.fields_pooling:
          dim += FLAGS.other_emb_dim

      if FLAGS.use_cold_emb:
        self.cold_emb = nn.Embedding(10, FLAGS.other_emb_dim)
        num_others += 1
        if 'concat' in FLAGS.fields_pooling:
          dim += FLAGS.other_emb_dim

    if FLAGS.fields_pooling:
      self.fields_pooling = le.layers.Pooling(FLAGS.fields_pooling)
      if 'concat' in FLAGS.fields_pooling:
        self.fields_pooling.set_output_size(dim, 'concat')
      if 'dot' in  FLAGS.fields_pooling:
        # TODO not consider dense emb right now
        num_fea = self.num_fields + num_others
        dot_dim = (num_fea * (num_fea - 1)) // 2
        # dot_dim = num_fea ** 2
        self.fields_pooling.set_output_size(dot_dim, 'dot')
      dim = self.fields_pooling.output_size

    if not FLAGS.mlp_dims:
      self.mlp = None
    else:
      dims = [int(x) for x in FLAGS.mlp_dims.split(',')]
      activation = FLAGS.dense_activation
      assert activation[0].isupper(), activation
      self.mlp = le.layers.MLP(dim, dims, dropout=FLAGS.mlp_drop, activation=activation)
      dim = self.mlp.output_dim
      
    if FLAGS.multi_obj_type == 'shared_bottom':
      if FLAGS.use_task_mlp:
        dims = [int(x) for x in FLAGS.task_mlp_dims.split(',')]
        self.task_mlp = le.layers.MLP(dim, dims, activation=activation, dropout=FLAGS.mlp_drop)
        self.task_mlp2 = le.layers.MLP(dim, dims, activation=activation, dropout=FLAGS.mlp_drop)
        dim = self.task_mlp.output_dim        

    if FLAGS.fields_pooling_after_mlp:
      self.fields_pooling_after_mlp = le.layers.Pooling(FLAGS.fields_pooling_after_mlp, input_size=FLAGS.hidden_size)
      dim += self.fields_pooling_after_mlp.output_size 

    if FLAGS.deep_out_dim == 1:
      self.dense = nn.Linear(dim, 1)
      if FLAGS.multi_obj_type:
        self.dense2 = nn.Linear(dim, 1)
      dim = 1

    self.odim = dim   

  def build(self):
    self.emb.build()
    self.doc_emb.build()
    self.user_emb.build()
      
  def deal_others(self, input):
    other_embs = []
    
    if not FLAGS.use_other_embs:
      return other_embs 

    if FLAGS.use_user_emb:
      # [bs, 1, dim] -> [bs, dim]
      other_embs += [self.user_emb(input['uid']).squeeze(1)]

    if FLAGS.use_doc_emb:
      other_embs += [self.doc_emb(input['did']).squeeze(1)]

    if FLAGS.use_history_emb:
      histories = input['history']
     
      if not 'Match' in FLAGS.hpooling:
        mask = None if FLAGS.hpooling in ['sum', 'mean'] else histories.eq(0)
        # [bs, dim]
        x_hist = self.hpooling(self.doc_emb(histories), mask)
      else:
        pass
      other_embs += [x_hist]

    if FLAGS.use_time_emb:
      time_interval = input['time_interval']
      if FLAGS.time_bins_per_day:
        num_bins = FLAGS.time_bins_per_hour * 24
        num_large_bins = FLAGS.time_bins_per_day
        intervals_per_large_bin = int(num_bins / num_large_bins)
        tmask = (time_interval > 1).int()
        tbase = time_interval * (1 - tmask)
        time_interval_large = (((time_interval - 2 - FLAGS.time_bin_shift_hours * FLAGS.time_bins_per_hour) % num_bins)/ intervals_per_large_bin).int() + 2
        time_interval_large = tbase + time_interval_large * tmask
        x_time = self.time_emb(time_interval_large)
      else:
        x_time = self.time_emb(time_interval)

      time_weekday = input['time_weekday'] 
      x_weekday = self.weekday_emb(time_weekday)
      other_embs += [x_time.squeeze(1), x_weekday.squeeze(1)]      

    if FLAGS.use_timespan_emb:
      timespan_interval = input['timespan_interval']
      x_timespan = self.timespan_emb(timespan_interval)
      other_embs += [x_timespan.squeeze(1)]
    
    if FLAGS.use_product_emb:
      x_product = self.product_emb(input['product_id'])
      other_embs += [x_product.squeeze(1)]

    if FLAGS.use_cold_emb:
      cold = input['cold'] 
      x_cold = self.cold_emb(cold) 
      other_embs += [x_cold.squeeze(1)]    

    return other_embs
    
  def deal_onehot_v1(self, indexes, values, fields):
    x = self.emb(indexes)

    if FLAGS.field_emb:
      x = torch.cat([x, self.field_emb(fields)], -1)

    values = values.unsqueeze(-1)
    x = x * values
    
    if not FLAGS.onehot_fields_pooling:
      mask = None if FLAGS.pooling == 'sum' else indexes.eq(0)
      x = self.pooling(x, mask)
      xs = x.unsqueeze(1)
    else:
      x = le.unsorted_segment_sum(x, fields, self.num_fields + 1)
      x = x[:, 1:, :]
      xs = x
    
    return xs

  def forward(self, input):
    indexes = input['index']
    values = input['value']
    fields = input['field']

    # if gezi.is_valid_step():
    #   mask = indexes.ne(0).float()
    #   mean_len = torch.mean(torch.sum(mask, 1))
    #   max_len = torch.max(torch.sum(mask, 1))
    #   min_len = torch.min(torch.sum(mask, 1))

    #   gezi.summary_scalar('mean_len', mean_len)
    #   gezi.summary_scalar('max_len', max_len)
    #   gezi.summary_scalar('min_len', min_len)
    #   logging.info('abc', gezi.global_step(), max_len.detach().cpu(), indexes.shape[1])

    if not FLAGS.use_deep_val:
      values = input['binary_value']
  
    x_all = []
    other_embs = self.deal_others(input)
    if other_embs:
      assert FLAGS.fields_pooling
      x_all += [torch.stack(other_embs, 1)]
    
    if FLAGS.use_onehot_emb:
      if not isinstance(fields, dict):
        xs = self.deal_onehot_v1(indexes, values, fields)
      else:
        xs = self.emb(indexes, values)
      x_all += [xs]

    assert x_all
    x_all = torch.cat(x_all, 1)

    if FLAGS.fields_pooling:
      x = self.fields_pooling(x_all)

    if self.mlp:
      x = self.mlp(x)
      
    if FLAGS.multi_obj_type and FLAGS.use_task_mlp:
      x2 = self.task_mlp2(x)
      x = self.task_mlp(x)
      
    if FLAGS.fields_pooling_after_mlp:
      x_after_mlp = self.fields_pooling_after_mlp(x_all)
      x = torch.cat([x, x_after_mlp], -1)
      if FLAGS.multi_obj_type:
        x2 = torch.cat([x2, x_after_mlp], -1)

    if FLAGS.deep_out_dim == 1:
      x = self.dense(x)
      if FLAGS.multi_obj_type:
        x2 = self.dense2(x2)
    
    if FLAGS.multi_obj_type:
      self.x2 = x2
    return x

class Model(nn.Module):   
  def __init__(self):
    super(Model, self).__init__()

    Embedding = get_embedding

    self.user_emb = Embedding('user')
    self.doc_emb = Embedding('doc')

    self.day_emb = Embedding('day', 7)
    self.device_emb = Embedding('device', 3)
    self.author_emb = Embedding('author')
    self.song_emb = Embedding('song')
    self.singer_emb = Embedding('singer')
    #注意如果下面没有用self.video_time_emb 那么是没有build这个layer print会失败
    self.video_time_emb = Embedding('video_time', 70)
    self.video_time2_emb = Embedding('video_time2', 10)
    self.is_first_emb = Embedding('is_first', 2)

    self.position_emb = Embedding('position', 100)
    self.span_emb = Embedding('span', 20)

    doc_lookup_file = '../input/doc_lookup.npy' if FLAGS.rare_unk else '../input/doc_ori_lookup.npy'
    logging.info('doc_lookup_file', doc_lookup_file)
    doc_lookup_npy = np.load(doc_lookup_file)
    # TO int
    self.doc_lookup = nn.Embedding.from_pretrained(doc_lookup_npy, freeze=True)

    # self.doc_lookup = mt.layers.LookupArray(doc_lookup_file, name='doc_lookup')

    # self.feed_emb = mt.layers.LookupArray('../input/feed_embeddings.npy', name='feed_emb')
    self.feed_emb = Embedding('feed')

    self.num_actions_emb = Embedding('num_actions', 100)
    self.num_read_comments_emb = Embedding('num_read_comments', 100)
    self.num_comments_emb = Embedding('num_comments', 100)
    self.num_likes_emb = Embedding('num_likes', 100)
    self.num_click_avatars_emb = Embedding('num_click_avatars', 100)
    self.num_forwards_emb = Embedding('num_forwards', 100)
    self.num_follows_emb = Embedding('num_follows', 100)
    self.num_favorites_emb = Embedding('num_favorites', 100)

    self.fresh_emb = Embedding('fresh', 20)

    self.embs = {
      'user': self.user_emb,
      'doc': self.doc_emb,
      'day': self.day_emb,
      'device': self.device_emb,
      'author': self.author_emb,
      'song': self.song_emb,
      'singer': self.singer_emb,
      'video_time': self.video_time_emb,
      'video_time2': self.video_time2_emb,
      'feed': self.feed_emb,
      'is_first': self.is_first_emb,
      'num_actions': self.num_actions_emb,
      'num_read_comments': self.num_read_comments_emb,
      'num_comments': self.num_comments_emb,
      'num_likes': self.num_likes_emb,
      'num_click_avatars': self.num_click_avatars_emb,
      'num_forwards': self.num_forwards_emb,
      'num_follows': self.num_follows_emb,
      'num_favorites': self.num_favorites_emb,
      'fresh': self.fresh_emb,
    }

    self.tag_emb = Embedding('tag')
    self.key_emb = Embedding('key')

    self.char_emb = Embedding('char')
    self.word_emb = Embedding('word')

    dim = 0
    for feat in FLAGS.feats:
      if feat == 'doc' and not FLAGS.use_doc_emb:
        continue
      if feat == 'user' and not FLAGS.use_user_emb:
        continue
      dim += FLAGS.emb_dim

    self.pooling = le.layers.Pooling(FLAGS.pooling)
    dim = self.pooling.output_size

    dims = FLAGS.mlp_dims[:2]
    self.mlp = le.layers.MLP(dims, dropout=FLAGS.dropout)
    dims2 = [dims[0], dims[1], FLAGS.emb_dim]
    self.dense_mlp = le.layers.MLP(dims2, dropout=FLAGS.dropout)
    self.mlps = [] 
    self.denses = []
    for action in FLAGS.loss_list:
      self.mlps.append(mt.layers.MLP([512, FLAGS.emb_dim], drop_rate=FLAGS.dropout, name=f'mlp_{action}'))
      self.denses.append(nn.Linear(FLAGS1))

    self.dense = nn.Linear(dim, len(FLAGS.loss_list))

    le.keras_init(self, keras_emb=True, keras_linear=True)


  def forward(self, input):


      
    self.logit = y

    return y

