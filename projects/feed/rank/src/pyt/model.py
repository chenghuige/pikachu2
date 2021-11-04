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
import lele

import numpy as np
from projects.feed.rank.src.config import *
from projects.feed.rank.src.pyt import util 

class Wide(nn.Module):
  def __init__(self):
    super(Wide, self).__init__()
    HashEmbedding = getattr(lele.layers.hash_embedding, FLAGS.hash_embedding_type)
    ## here unlike Deep.emb sparse will be much slower
    if not gezi.get_global('embedding_keys'):
      self.emb = HashEmbedding(FLAGS.wide_feature_dict_size, 1, sparse=False, padding_idx=0, 
                              num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner)
    else:
      self.emb = lele.layers.EmbeddingBags(FLAGS.wide_feature_dict_size, 1, sparse=False, padding_idx=0,
                                    num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner,
                                    Embedding=HashEmbedding, split=FLAGS.split_embedding)
    # self.emb = nn.EmbeddingBag(FLAGS.feature_dict_size, 1, sparse=FLAGS.sparse_emb, mode='sum')
    # self.emb = lele.Embedding(FLAGS.feature_dict_size + 1, 1, sparse=FLAGS.sparse_emb)

    #self.bias = torch.zeros(1, requires_grad=True).cuda()
    # https://discuss.torch.rch.org/t/tensors-are-on-different-gpus/1450/28 
    # without below multiple gpu will fail
    # RuntimeError: binary_op(): expected both inputs to be on same device, but input a is on cuda:0 and input b is on cuda:7 
    self.bias = nn.Parameter(torch.zeros(1))

    embedding_keys = gezi.get_global('embedding_keys')
    self.num_fields = FLAGS.field_dict_size - 1 if not embedding_keys else len(embedding_keys)
    logging.debug('wide num_fields:', self.num_fields)
    self.odim = 1 if not FLAGS.use_fm_first_order else self.num_fields 

  def build(self):
    pass

  def deal_v1(self, indexes, values, fields):
    x = self.emb(indexes)
    x = x.squeeze(-1)
    x = x * values

    if FLAGS.use_fm_first_order:
      x = x.unsqueeze(-1)
      # num_fields + 1 for padding 0
      x = lele.unsorted_segment_sum(x, fields, self.num_fields + 1)
      # [None, F]
      x = x.view(-1, (self.num_fields + 1))
      # ignore first field as 0 is padding purpose
      x = x[:,1:]
    else:
      x = x.sum(1, keepdim=True)
      
    return x

  def forward(self, input):
    indexes = input['index']
    values = input['value']
    fields = input['field']

    if not FLAGS.use_wide_val:
      values = input['binary_value']

    if not isinstance(fields, dict):
        x = self.deal_v1(indexes, values, fields)
    else:
      x = self.emb(indexes, values)
      x = torch.sum(x, 1)
    
    # x = self.emb(indexes, per_sample_weights=values)
    if not FLAGS.use_fm_first_order:
      x = x + self.bias
    return x  

class Deep(nn.Module):
  def __init__(self):
    super(Deep, self).__init__()
  
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()
    kwargs = dict(padding_idx=0, sparse=FLAGS.sparse_emb, num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner)
    if FLAGS.use_onehot_emb:
      if not gezi.get_global('embedding_keys'):
        self.emb = HashEmbedding(FLAGS.feature_dict_size, FLAGS.hidden_size, large_emb=FLAGS.large_emb, **kwargs)
      else:
        self.emb = lele.layers.EmbeddingBags(FLAGS.feature_dict_size, FLAGS.hidden_size, 
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
      self.pooling = lele.layers.Pooling(FLAGS.pooling)  

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
        self.hpooling = lele.layers.Pooling(FLAGS.hpooling)
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
      self.fields_pooling = lele.layers.Pooling(FLAGS.fields_pooling)
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
      self.mlp = lele.layers.MLP(dim, dims, dropout=FLAGS.mlp_drop, activation=activation)
      dim = self.mlp.output_dim
      
    if FLAGS.multi_obj_type == 'shared_bottom':
      if FLAGS.use_task_mlp:
        dims = [int(x) for x in FLAGS.task_mlp_dims.split(',')]
        self.task_mlp = lele.layers.MLP(dim, dims, activation=activation, dropout=FLAGS.mlp_drop)
        self.task_mlp2 = lele.layers.MLP(dim, dims, activation=activation, dropout=FLAGS.mlp_drop)
        dim = self.task_mlp.output_dim        

    if FLAGS.fields_pooling_after_mlp:
      self.fields_pooling_after_mlp = lele.layers.Pooling(FLAGS.fields_pooling_after_mlp, input_size=FLAGS.hidden_size)
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
      x = lele.unsorted_segment_sum(x, fields, self.num_fields + 1)
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

class WideDeep(nn.Module):   
  def __init__(self):
    super(WideDeep, self).__init__()
    
    if FLAGS.need_field_lookup:
      self.lookup_array = util.get_lookup_array()

    odim = 0
    if not FLAGS.deep_only:
      self.wide = Wide()
      if FLAGS.multi_obj_type:
        self.wide2 = Wide()
      odim += self.wide.odim
    if not FLAGS.wide_only:
      self.deep = Deep() 
      odim += self.deep.odim
          
    if odim > 1:
      idim = odim
      self.dense = nn.Linear(idim, 1)
      if FLAGS.multi_obj_type:
        self.dense2 = nn.Linear(idim, 1)
    else:
      self.dense = None

    lele.keras_init(self, FLAGS.keras_emb, FLAGS.keras_linear)

  def build(self):
    self.wide.build()
    self.deep.build()

  def adjust_input(self, input):
    if 'visited' not in input:
      input['visited'] = True

      if FLAGS.need_field_lookup:
        if not isinstance(input['field'], dict):
          # NOTICE you can only look up onece as now you need to first %  then lookup
          input['field'] = self.lookup_array(input['field'])

      if FLAGS.masked_fields:
        if not isinstance(input['field'], dict):
          assert FLAGS.need_field_lookup
          mask = input['field'].ne(0)
          input['index'] *= mask.long()
          input['value'] *= mask.float()
      
      if not FLAGS.ignore_zero_value_feat:
        if not isinstance(input['field'], dict):
          mask = (input['index'] == 0.).float()
          input['value'] += ((input['value'] == 0.).float() - mask)

      if (not FLAGS.use_deep_val) or (not FLAGS.use_wide_val):
        if not isinstance(input['value'], dict):
          input['binary_value'] = (input['value'] != 0).float()
        else:
          input['binary_value'] = {}
          for key in input['value']:
            input['binary_value'][key] = (input['value'][key] != 0).float()

      # TODO add field mask

  def forward(self, input):
    self.adjust_input(input)

    w = None
    if not FLAGS.deep_only:
      w = self.wide(input)
    
    d = None
    if not FLAGS.wide_only:
      d = self.deep(input)
      
    if w is not None and d is not None:
      x = torch.cat([w, d], 1)
    else:
      x = w if w is not None else d

    if self.dense is not None:
      y = self.dense(x)
    
    self.y_click = y
    self.prob = None
    if FLAGS.multi_obj_type:
      if not FLAGS.deep_only:
        w2 = self.wide2(input)
        self.w2 = w2
      if not FLAGS.wide_only:
        d2 = self.deep.x2
        self.d2 = d2
      if FLAGS.wide_only:
        x2 = w2
      elif FLAGS.deep_only:
        x2 = d2
      else:
        x2 = torch.cat([w2, d2], 1)
      
      if self.dense is not None:
        x2 = self.dense2(x2)
      self.y_dur = x2  
    
      # Notice bad name here.. actually should use name as exponent not power
      click_power = float(FLAGS.click_power.split(',')[0])
      dur_power = float(FLAGS.dur_power.split(',')[0])

      if self.training:
        click_power = 1.
        dur_power = 1.

      dur_need_sigmoid = True
      if 'jump' in FLAGS.multi_obj_duration_loss:
        if not 'BCE' in FLAGS.jump_loss:
          dur_need_sigmoid = False
      elif not 'BCE' in FLAGS.multi_obj_duration_loss:
        dur_need_sigmoid =False  
      
      self.prob_click = torch.sigmoid(self.y_click)
      self.prob_dur = torch.sigmoid(self.y_dur) if dur_need_sigmoid else self.y_dur  
      y = (self.prob_click ** click_power) * (self.prob_dur ** dur_power)
      # y = y ** 0.5
      self.prob = y
      y = lele.prob2logit(y)    
      
    self.logit = y
    self.pred = y
    self.logit = y
    self.need_sigmoid = True
    
    if not self.training:
      if self.prob is None:
        y = torch.sigmoid(y)
        self.prob = y
      else:
        y = self.prob
      self.need_sigmoid = False

    self.num_tw_histories = lele.length(input['tw_history']) - 1
    self.num_vd_histories = lele.length(input['vd_history']) - 1

    return y

