#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2019-07-26 20:15:30.419843
#   \Description  TODO maybe input should be more flexible, signle feature, cross, cat, lianxu choumi
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
import melt as mt

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

from tensorflow import keras
from tensorflow.keras import backend as K

import numpy as np
from projects.feed.rank.src.config import *
from projects.feed.rank.src import util 
from projects.feed.rank.src import loss

from projects.feed.rank.src.others import *
from projects.feed.rank.src.history import *
from projects.feed.rank.src.others import *

import gezi 
logging = gezi.logging

# output logits!
class Wide(keras.Model):
  def __init__(self):
    super(Wide, self).__init__()
    self.HashEmbedding = getattr(mt.layers, FLAGS.hash_embedding_type)      
    # self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)
    self.regularizer = None
    self.combiner = FLAGS.hash_combiner if FLAGS.hash_combiner != 'concat' else 'sum'

    if FLAGS.use_wide_position_emb:
      self.pos_emb = Embedding(FLAGS.num_positions, 1, name='pos_emb')

    if FLAGS.visual_emb:
      mt.visualize_embedding(self.emb, os.path.join(FLAGS.data_dir, 'feature.project'))
      # mt.histogram_summary(self.emb, 'wide.emb')

  # put bias in build so we can track it as WideDeep/Wide/bias
  def build(self, input_shape):
    if not gezi.get_global('embedding_keys'):
      self.emb = self.HashEmbedding(FLAGS.wide_feature_dict_size, 1, num_buckets=FLAGS.num_feature_buckets, 
                              embeddings_regularizer=self.regularizer, need_mod=True, 
                              combiner=self.combiner, name='emb/emb')
    else:
      self.emb = mt.layers.EmbeddingBags(FLAGS.wide_feature_dict_size, 1, num_buckets=FLAGS.num_feature_buckets, 
                                          Embedding=self.HashEmbedding, embeddings_regularizer=self.regularizer,
                                          split=FLAGS.split_embedding, combiner=self.combiner, name='emb')
    embedding_keys = gezi.get_global('embedding_keys')
    self.num_fields = FLAGS.field_dict_size if not embedding_keys else len(embedding_keys)
    logging.debug('wide num_fields:', self.num_fields)
    
    self.bias = self.add_weight(name='bias',
                                shape=(1,),
                                initializer='zeros',
                                # regularizer=self.regularizer,
                                dtype=tf.float32,
                                trainable=True)
    self.built = True

  def deal_v1(self, indexes, values, fields):
    infer = FLAGS.is_infer
    if FLAGS.sparse_to_dense:
      num_fields = self.num_fields
      if infer:
        x = self.emb(indexes)
      else:
        x = self.emb(indexes)
      x = K.squeeze(x, -1)
      x = x * values
      
      if FLAGS.use_fm_first_order:
        x = tf.expand_dims(x, -1)
        x = mt.unsorted_segment_embs(x, fields, num_fields, combiner=FLAGS.pooling)
        # [None, F]
        x = K.reshape(x, [-1, num_fields])
        # ignore first field as 0 is padding purpose
        x = x[:,1:]
      else:
        self.w = x 
        self.v = values
        x = K.sum(x, 1, keepdims=True)
    else:
      x = tf.nn.embedding_lookup_sparse(params=self.emb(None), sp_ids=indexes, sp_weights=values, combiner='sum') 
    return x

  def call(self, input):
    """outputs is [batch_size, 1]"""
    infer = FLAGS.is_infer
    indexes = input['index']
    fields = input['field']
    values = input['value']  
    if not FLAGS.use_wide_val:
      values = input['binary_value']
    if infer:
      uindexes = input['user_index']
      uvalues = input['user_value']
      ufields = input['user_field']
    
    if not isinstance(fields, dict):
      x = self.deal_v1(indexes, values, fields)
    else:
      x = self.emb(indexes, values)
      x = K.sum(x, 1)

    if not FLAGS.use_fm_first_order:
      x = x + self.bias
    return x  

# TODO try bottom mlp as drlm
class Deep(mt.Model):
  def __init__(self):
    super(Deep, self).__init__()

    # self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)
    self.regularizer = None
    Embedding = keras.layers.Embedding
    SimpleEmbedding = mt.layers.SimpleEmbedding
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()
    kwargs = dict(num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner, 
                  embeddings_regularizer=self.regularizer, num_shards=FLAGS.num_shards)
    self.kwargs = kwargs
    self.HashEmbedding = HashEmbedding

    if FLAGS.visual_emb:
      mt.visualize_embedding(self.emb, os.path.join(FLAGS.data_dir, 'feature.project'))
      # mt.histogram_summary(self.emb, 'deep.emb')
    self.emb_dim = FLAGS.hidden_size

    self.emb_activation = None
    if FLAGS.emb_activation:
      self.emb_activation = keras.layers.Activation(FLAGS.emb_activation)

    self.model_others = Others()

    if not FLAGS.mlp_dims:
      self.mlp = None
    else:
      dims = [int(x) for x in FLAGS.mlp_dims.split(',')]
      activation = FLAGS.dense_activation if not FLAGS.mlp_norm else None
      drop_rate = FLAGS.mlp_drop if not FLAGS.mlp_norm else None
      self.mlp = mt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
      if FLAGS.use_residual:
        self.residual = mt.layers.Residual(dims[-1], 2)
      if FLAGS.use_dense_feats:
        bot_dims = [int(x) for x in FLAGS.bot_mlp_dims.split(',')]
        self.bottom_mlp = mt.layers.MLP(bot_dims, activation=activation,
            drop_rate=drop_rate)
      if FLAGS.multi_obj_type == 'shared_bottom':
        self.mlp2 = mt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
        if FLAGS.use_residual:
          self.residual2 = mt.layers.Residual(dims[-1], 2)
        self.bottom_mlp2 = mt.layers.MLP(dims, activation=activation,
            drop_rate=drop_rate)

    if FLAGS.mmoe_dims:
      dims = [int(x) for x in FLAGS.mmoe_dims.split(',')]
      self.mmoe = mt.layers.MMoE(2, FLAGS.num_experts, dims)

    if FLAGS.use_task_mlp:
      dims = [int(x) for x in FLAGS.task_mlp_dims.split(',')]
      self.task_mlp = mt.layers.MLP(dims, activation=activation,
        drop_rate=drop_rate)
      self.task_mlp2 = mt.layers.MLP(dims, activation=activation,
        drop_rate=drop_rate)

      # bad result
      if FLAGS.mlp_norm:
        self.batch_norm = tf.keras.layers.BatchNormalization()  

    act = FLAGS.dense_activation if FLAGS.deep_final_act else None    

    self.pooling = mt.layers.Pooling(FLAGS.pooling)
    self.sum_pooling = mt.layers.Pooling('sum')

    if FLAGS.fields_pooling:
      self.fields_pooling = mt.layers.Pooling(FLAGS.fields_pooling)
    
    if FLAGS.multi_fields_pooling:
      self.multi_fields_pooling = mt.layers.Pooling(FLAGS.multi_fields_pooling)
      self.multi_fields_pooling2 = mt.layers.Pooling(FLAGS.multi_fields_pooling)

    if FLAGS.fields_pooling_after_mlp:
      self.fields_pooling_after_mlp = mt.layers.Pooling(FLAGS.fields_pooling_after_mlp)
    if FLAGS.fields_pooling_after_mlp2:
      self.fields_pooling_after_mlp2 = mt.layers.Pooling(FLAGS.fields_pooling_after_mlp2)

    if FLAGS.other_emb_dim != self.emb_dim:
      self.dense_to_other = keras.layers.Dense(FLAGS.other_emb_dim)
    else:
      self.dense_to_other = None

    if FLAGS.emb_drop:
      self.dropout = keras.layers.Dropout(FLAGS.emb_drop)

    if FLAGS.deep_out_dim == 1:
      self.dense = keras.layers.Dense(1)
      if FLAGS.multi_obj_type:
        self.dense2 = keras.layers.Dense(1)

  def build(self, input_shape):
    if not gezi.get_global('embedding_keys'):
      self.emb = self.HashEmbedding(FLAGS.feature_dict_size, self.emb_dim, name='emb/emb', **self.kwargs)
    else:
      self.emb = mt.layers.EmbeddingBags(FLAGS.feature_dict_size, self.emb_dim, 
          Embedding=self.HashEmbedding, split=FLAGS.split_embedding, name='emb', **self.kwargs)
    embedding_keys = gezi.get_global('embedding_keys')
    
    if not gezi.get_global('embedding_keys'):
      # Notice here is actually before self.emb.build.. so not correct num_fields for EmbeddingBags which might change embedding_keys
      self.num_fields = FLAGS.field_dict_size if not embedding_keys else len(embedding_keys)
      logging.debug('deep num_fields:', self.num_fields)

    if FLAGS.emb_activation:
      self.bias = K.variable(value=[0.], name='bias')
    self.built = True

  def deal_onehot_v1(self, indexes, values, fields):
    self.fm = None
    num_fields = self.num_fields
    if not FLAGS.sparse_to_dense:    
      assert FLAGS.pooling == 'sum' or FLAGS.pooling == 'mean'
      assert not FLAGS.use_fields, "TODO.."
      values_ = values if FLAGS.use_deep_val else None
      with mt.device(FLAGS.emb_device):
        x = tf.nn.embedding_lookup_sparse(params=self.emb(None), sp_ids=indexes, sp_weights=values_, combiner=FLAGS.pooling)
      if FLAGS.field_emb:
        x = K.concatenate([x, tf.nn.embedding_lookup_sparse(params=self.field_emb(None), sp_ids=fields, sp_weights=None, combiner=FLAGS.pooling)], axis=-1) 
      return x
    else:
      x_len = mt.length(indexes) 
      with mt.device(FLAGS.emb_device):
        if FLAGS.is_infer:
          # x = tf.cond(util.is_null(uindexes), lambda: self.emb(indexes), lambda: util.merge_embs(self.emb, uindexes, pre_indexes))
          x = self.emb(indexes)
        else:
          x = self.emb(indexes)

      if FLAGS.field_emb:
        x_field = self.field_emb(fields)
        if FLAGS.disable_field_emb:
          x_field = tf.stop_gradient(x_field)
        x = K.concatenate([x, x_field])

      values = K.expand_dims(values, -1)
      x = x * values

      num_fields_ = 1
      if FLAGS.onehot_fields_pooling:
        num_fields_ = num_fields - 1
        # x = tf.math.unsorted_segment_sum(x, fields, num_fields)  
        # like [batch_size * max_dim, hidden_dim] ->   [batch_size * num_segs, hidden_dim] -> [batch_size, num_segs, hidden_dim]
        # TODO mask and filter out zero embeddings 
        x = mt.unsorted_segment_embs(x, fields, num_fields, combiner=FLAGS.pooling)
        # x = mt.unsorted_segment_sum(x, fields, num_fields)
        # TODO do not need reshape ? change unsorted ..
        x = x[:, 1:, :]
        xs = x
      else:
        # if not do field_concat we can do fm with other_embs with only one onehot emb output
        # for sum since x * values already do masked pooling
        if FLAGS.pooling == 'allsum' or FLAGS.pooling == 'sum':
          x = K.sum(x, 1)
        elif FLAGS.pooling == 'mean':
          x = K.mean(x, 1)
        else:
          assert FLAGS.index_addone, 'can not calc length for like 0,1,2,0,0,0'
          x = self.pooling(x, x_len)
        xs = tf.expand_dims(x, 1)

    return xs

  def call(self, input):
    """outputs is [batch_size, 1]"""
    self.clear()

    infer = FLAGS.is_infer
    indexes = input['index']
    fields = input['field']
    values = input['value']  
    if not FLAGS.use_deep_val:
      values = input['binary_value']  
    if infer:
      uindexes = input['user_index']
      uvalues = input['user_value']
      ufields = input['user_field']

    dense_feats = input['dense'] if FLAGS.use_dense_feats else None
    
    if 'x_all' not in input:
      x_all = []

      other_embs = self.model_others(input)
      logging.debug('-----------other_embs', len(other_embs))
      self.merge(self.model_others.feats)
      if other_embs:
        other_emb = tf.stack(other_embs, 1)
        x_all += [other_emb]

      if dense_feats is not None:
        x_dense = self.bottom_mlp(dense_feats)
        x_dense = tf.expand_dims(x_dense, 1)
        x_all += [x_dense]
        self.add(x_dense, 'dense')

      # just use sparse_to_dense 
      if FLAGS.use_onehot_emb:
        # xs is before pooling, x is concat pooling
        if not isinstance(fields, dict):
          xs = self.deal_onehot_v1(indexes, values, fields)
        else:
          xs = self.emb(indexes, values)

        for feat in gezi.get('fields'):
          self.add(None, feat)
      
        if self.dense_to_other is not None:
          xs = self.dense_to_other(xs)

        logging.debug('-----------onehot_embs', xs.shape)
        x_all += [xs]

      # if FLAGS.use_label_emb:
      #   x_label = self.label_emb(tf.zeros_like(input['did']))
      #   x_all = [x_label, *x_all]
          
      assert x_all
      x_all = tf.concat(x_all, 1)
      self.dump_feats()

      if FLAGS.fields_norm:
        x_all = tf.math.l2_normalize(x_all, -1)

      if FLAGS.transform:
        x_all = self.transformer(x_all) 
        # x = tf.concat([x_all2[:, 0, :], x], 1)
    else:
      x_all = input['x_all']

    self.x_all = x_all
    # logging.info('deep final (onehot+other) num_fields:', mt.get_shape(x_all, 1))
    if FLAGS.fields_pooling:
      x = self.fields_pooling(x_all) 

    if FLAGS.udh_concat:
      x = tf.concat([x_all[:, 0, :], x_all[:, 1, :], x_all[:, 2, :], x], axis=-1)

    if dense_feats is not None:
      x = tf.concat([x_dense, x], 1)

    if FLAGS.concat_count > 0:
      x = tf.concat([x, *other_embs[:FLAGS.concat_count]], 1)

    x2 = None      
    # TODO better support dense feats with feature intersection
    # if dense_feats:
    #   dense_feats =K.concatenate(dense_feats)
    #   dx = self.bottom_mlp(dense_feats)
    #   x = K.concatenate([x, dx])
    #   # if FLAGS.multi_obj_type:
    #   #   dx2 = self.bottom_mlp2(dense_feats)
    #   #   x2 = K.concatenate([x2, dx2])

    self.before_mlp = x if 'before_mlp' not in input else input['before_mlp']
  
    if self.mlp:
      x = self.mlp(x)
      if FLAGS.use_residual:
        x = self.residual(x)
      if FLAGS.multi_obj_type:
        # if share mlp not need to do mlp2 but for old models compat here
        if (not FLAGS.multi_obj_share_mlp) or FLAGS.compat_old_model:
          x2 = self.mlp2(x2)
          x2 = self.residual2(x2)

    self.after_mlp = x

    if FLAGS.mlp_norm:
      x = self.batch_norm(x)

    if FLAGS.mmoe_dims:
      x, x2 = self.mmoe(x)

    if FLAGS.multi_fields_pooling:
      assert FLAGS.multi_obj_type 
      mx2 = self.multi_fields_pooling2(x_all)
      if x2 is None:
        x2 = x
      x2 = K.concatenate([x2, mx2])
      mx = self.multi_fields_pooling(x_all)
      x =  K.concatenate([x, mx])
  
    if FLAGS.use_task_mlp:
      assert FLAGS.multi_obj_type 
      assert FLAGS.multi_obj_share_mlp
      if x2 is None:
        x2 = x
      x2 = self.task_mlp2(x2)   
      x = self.task_mlp(x)

    if FLAGS.multi_obj_type and x2 is None:
      x2 = x

    if FLAGS.fields_pooling_after_mlp:
      x_after_mlp = self.fields_pooling_after_mlp(x_all)
      x = K.concatenate([x, x_after_mlp])
      if FLAGS.multi_obj_type:
        if FLAGS.fields_pooling_after_mlp2:
          x_after_mlp = self.fields_pooling_after_mlp2(x_all)
        x2 = K.concatenate([x2, x_after_mlp])

      # if FLAGS.use_deep_position_emb:
      #   position = input['position']
      #   x_pos = self.pos_emb(position)
      #   x_pos = self.pos_mlp(x_pos)
      #   x_pos = self.pos_dense(x_pos)
      #   self.y_pos = x_pos
      # def merge_pos(x, x_pos):
      #   if FLAGS.position_combiner == 'concat':
      #     x = K.concatenate([x, x_pos])
      #   elif FLAGS.position_combiner == 'add':
      #     x = x + x_pos
      #   else:
      #     raise ValueError('Unsuported position_combiner %s' % FLAGS.position_combiner)
      # x = merge_pos(x, x_pos)
      # if FLAGS.multi_obj_type:
      #   x2 = merge_pos(x2, x_pos)

    self.vec = x    
    self.click_feat = x
    if FLAGS.multi_obj_type:
      self.dur_feat = x2
      
    if FLAGS.deep_out_dim == 1:
      x = self.dense(x)
      if FLAGS.multi_obj_type:
        x2 = self.dense2(x2)

    if FLAGS.multi_obj_type:
      self.x2 = x2

    return x

class WideDeep(mt.Model):   
  def __init__(self, **kwargs):
    super(WideDeep, self).__init__(**kwargs)

    if not FLAGS.deep_only:
      self.wide = Wide()
      if FLAGS.multi_obj_type and FLAGS.multi_obj_type != 'simple':
        self.wide2 = Wide()
    if not FLAGS.wide_only:
      self.deep = Deep() 

    self.dense = keras.layers.Dense(1)
    if 'softmax' in FLAGS.multi_obj_duration_loss:
      self.dense_softmax = keras.layers.Dense(FLAGS.num_duration_classes)
    elif 'jump' in FLAGS.multi_obj_duration_loss:
      self.dense_dur = keras.layers.Dense(1)
      if 'dense' in FLAGS.multi_obj_duration_loss:
        self.dense_merge = keras.layers.Dense(1)

    if FLAGS.multi_obj_type and FLAGS.multi_obj_type != 'simple':
      self.dense2 = keras.layers.Dense(1)
      
    self.dur_infer_ratio = FLAGS.multi_obj_duration_infer_ratio

    lookup_arr = util.get_lookup_array()
    if lookup_arr is not None:
      self.lookup_field = mt.layers.LookupArray(lookup_arr, need_mod=True, name='field_array')

    if FLAGS.use_position_emb:
      Embedding = keras.layers.Embedding
      self.pos_emb = Embedding(FLAGS.num_positions, FLAGS.hidden_size, name='pos_emb')
      self.pos_mlp = mt.layers.MLP([8, 4], activation='relu')
      self.pos_dense = keras.layers.Dense(1)

    self.input_ = None

  def adjust_input(self, input):
    # input = input.copy()
    # 只做一次adjust_input
    if 'visited' not in input:
      input['visited'] = True
      if FLAGS.need_field_lookup:
        # NOTICE you can only look up onece as now you need to first %  then lookup
        if not isinstance(input['field'], dict):
          input['field'] = self.lookup_field(input['field'])

      input['value'] = tf.clip_by_value(input['value'], -10., 10.)

      # if FLAGS.masked_fields:
      #   if not isinstance(input['field'], dict):
      #     assert FLAGS.need_field_lookup
      mask = K.not_equal(input['field'], 0)
      input['index'] *= tf.cast(mask, input['index'].dtype)
      input['value'] *= tf.cast(mask, tf.float32)

      # print('-----index', input['index'])
      # print('-----field', input['field'])
      # print('-----value', input['value'])

      # TODO HACK now for input val problem we turn 0 value to 1.
      # NOTICE here padding value 0 is also turn to 1 if not mask by index
      if not FLAGS.ignore_zero_value_feat:
        if not isinstance(input['index'], dict):
          mask = tf.cast(K.equal(input['index'], 0), tf.float32)
          input['value'] += (tf.cast(K.equal(input['value'], 0.), tf.float32) - mask)
      
      if (not FLAGS.use_wide_val) or (not FLAGS.use_deep_val):
        if not isinstance(input['value'], dict):
          input['binary_value'] = tf.cast(K.not_equal(input['value'], 0), tf.float32)    
        else:
          input['binary_value'] = {}
          for key in input['value']:
            if isinstance(input['value'][key], tf.Tensor):
              input['binary_value'][key] = tf.cast(K.not_equal(input['value'][key], 0), tf.float32)
            else:
              input['binary_value'][key] = tf.sparse.SparseTensor(input['value'][key].indices,
                                                                  tf.cast(K.not_equal(input['value'][key].values, 0), tf.float32),
                                                                  input['value'][key].dense_shape)

      mt.try_squeeze(input, exclude_keys=['click', 'duration', 'weight'])

    return input

  # @tf.function  
  def call(self, input):
    """outputs is [batch_size, 1]"""
    self.clear()

    self.adjust_input(input)
    self.input_ = input

    if not FLAGS.deep_only:
      w = self.wide(input)
      self.w = w

    if not FLAGS.wide_only:
      d = self.deep(input)
      self.d = d

    if FLAGS.wide_only:
      x = w 
    elif FLAGS.deep_only: 
      x = d
    else:
      x = K.concatenate([w, d])
        
    if mt.get_shape(x, 1) > 1:
      x = self.dense(x)
            
    self.y_click = x   
    
    # TODO position embedding here for multi obj2 ?
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
        x2 = K.concatenate([w2, d2])
      if mt.get_shape(x2, 1) > 1:
        x2 = self.dense2(x2)
      self.y_dur = x2      
    y = self.y_click 
    
    self.prob = None
    
    if 'softmax' in FLAGS.multi_obj_duration_loss:
      y = tf.reduce_sum(tf.math.softmax(self.y_softmax) * tf.range(FLAGS.num_duration_classes, dtype=tf.float32), axis=1) + tf.math.sigmoid(self.y_click)
    else:    
      self.prob_click = tf.math.sigmoid(self.y_click)
      if FLAGS.finish_loss:
        self.prob_finish = tf.math.sigmoid(self.y_finish)

      if FLAGS.multi_obj_type:
        product = util.get_product_id(input['product'])

        click_powers = list(map(float, FLAGS.click_power.split(',')))
        dur_powers = list(map(float, FLAGS.dur_power.split(',')))

        click_power = tf.nn.embedding_lookup(tf.constant(click_powers), product)
        dur_power = tf.nn.embedding_lookup(tf.constant(dur_powers), product)

        cb_click_powers = list(map(float, FLAGS.cb_click_power.split(',')))
        cb_dur_powers = list(map(float, FLAGS.cb_dur_power.split(',')))
        cb_click_power = tf.nn.embedding_lookup(tf.constant(cb_click_powers), product)
        cb_dur_power = tf.nn.embedding_lookup(tf.constant(cb_dur_powers), product)     

        cb = tf.cast(util.is_cb_user(input['rea']), tf.float32)
        
        # Notice bad name here.. actually should use name as exponent not power
        click_power = click_power * (1.0 - cb) + cb_click_power * cb
        dur_power = dur_power * (1.0 - cb) + cb_dur_power * cb

        click_power = tf.reshape(click_power, tf.shape(y))
        dur_power = tf.reshape(dur_power, tf.shape(y))

        self.dur_need_sigmoid = True if 'cross_entropy' in FLAGS.multi_obj_duration_loss else False  
        
        self.prob_dur = tf.math.sigmoid(self.y_dur) if FLAGS.logit2prob else self.y_dur
        if K.learning_phase():
          click_power = 1.
          dur_power = 1.

        y = (self.prob_click ** click_power) * (self.prob_dur ** dur_power)
        if FLAGS.finish_loss:
          y *= self.prob_finish ** FLAGS.finish_power

        # this is the prob of final duration (normalized to 0-1)
        self.prob = y 
        # always here
        y = mt.prob2logit(y)

    # output shape is (batch_size, 1)
    self.pred = y
    self.logit = y
    self.need_sigmoid = True

    if FLAGS.use_position_emb:
      position = input['position']
      x_pos = self.pos_emb(position)
      x_pos = self.pos_mlp(x_pos)
      x_pos = self.pos_dense(x_pos)
      self.y_pos = x_pos

    # with NULL added -1 to 0
    self.num_tw_histories = mt.length(input['tw_history']) - 1
    self.num_vd_histories = mt.length(input['vd_history']) - 1

    self.merge(self.deep.feats)
    
    # TODO valid loss not ok ? though for evaluate.py output is ok
    if K.learning_phase() == 0:
      if self.prob is None:
        pred = tf.math.sigmoid(y)
        self.prob = pred
      else:
        pred = self.prob
      self.need_sigmoid = False
      # if FLAGS.debug_infer:
      #   pred = tf.cast(tf.cast(pred * 10000., tf.int32), tf.float32) / 10000. + 0.0000111 * FLAGS.model_mark
      # print(pred)
      return pred

    # 如果放到这里train没问题 但是valid 实际 input没有记录还是None NOTICE
    # self.input_ = input
    # tf.print(y, tf.math.sigmoid(y), input['click'], tf.compat.v1.losses.sigmoid_cross_entropy(input['click'], y))
    # tf.print(tf.compat.v1.losses.sigmoid_cross_entropy(input['click'], y))
    # tf.print(tf.shape(input['click']), tf.shape(y))
    return y

  def get_loss(self):
    loss_fn_ = loss.get_loss_fn(self)
    return self.loss_wrapper(loss_fn_)

  def init_predict(self):
    # TODO use decorator @is_infer
    K.set_learning_phase(0)
    FLAGS.is_infer = True
  
    if not tf.executing_eagerly():
      keys = gezi.get_global('embedding_keys')
      if not keys:
        index_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'index_feed')
        tf.compat.v1.add_to_collection('index_feed', index_feed)

        value_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0.]], dtype=tf.float32), [None, None], 'value_feed')
        tf.compat.v1.add_to_collection('value_feed', value_feed)

        field_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'field_feed')
        tf.compat.v1.add_to_collection('field_feed', field_feed)

        input = dict(index=index_feed, value=value_feed, field=field_feed)
      else:
        input = {}
        input['index'], input['value'], input['field'] = {},  {}, {}
        for key in keys:
          index_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], f'index_{key}_feed')
          tf.compat.v1.add_to_collection(f'index_{key}_feed', index_feed)
          input['index'][key] = index_feed

          value_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0.]], dtype=tf.float32), [None, None], f'value_{key}_feed')
          tf.compat.v1.add_to_collection(f'value_{key}_feed', value_feed)
          input['value'][key] = value_feed
        
      dummy = tf.zeros_like(index_feed[:, 0])
      input['position'] = dummy

      dummy = tf.expand_dims(dummy, 1)

      if FLAGS.use_dense_feats:
        num_dense  = gezi.get_global('num_dense')
        input['dense'] = tf.compat.v1.placeholder_with_default(tf.constant([[0.] * num_dense], dtype=tf.float32), [None, num_dense], 'dense_feed') # TODO

      input['user_index'] = tf.compat.v1.placeholder_with_default(tf.constant([0], dtype=tf.int64), [None], 'user_index_feed')
      tf.compat.v1.add_to_collection('user_index_feed', input['user_index'])

      input['user_value'] = tf.compat.v1.placeholder_with_default(tf.constant([0.], dtype=tf.float32), [None], 'user_value_feed')
      tf.compat.v1.add_to_collection('user_value_feed', input['user_value'])

      input['user_field'] = tf.compat.v1.placeholder_with_default(tf.constant([0], dtype=tf.int64), [None], 'user_field_feed')
      tf.compat.v1.add_to_collection('user_field_feed', input['user_field'])

      self.deep.model_others.init_predict(input, dummy)

      def _add(name, dtype=tf.int64):
        input[name] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=dtype), [None, 1], '%s_feed' % name)
        tf.compat.v1.add_to_collection('%s_feed' % name, input[name])
        input[name] += dummy
        # input[name] = tf.squeeze(input[name], 1)
      
      def _adds(names, dtype=tf.int64):
        for name in names:
          _add(name, dtype)

      # Notice we use time_interval to judge online infer or not when online infer time interval is one of the input but for offline using infer.py not
      names = [
        'product', 'time_interval', 'time_weekday', 'timespan_interval', 'device_info', 'rea', 
        'network', 'user_active', 'impression_count', 'impression_count_per_day',
        'coldstart_refresh_num', 'today_refresh_num', 'video_time', 'type',
        'koi'  
      ]

      _adds(names, tf.int64)

      # TODO just for debug
      tf.compat.v1.add_to_collection('dummy', dummy)
      tf.compat.v1.add_to_collection('uid', input['uid'])
      tf.compat.v1.add_to_collection('product', input['product'])
      tf.compat.v1.add_to_collection('time_interval', input['time_interval'])
      tf.compat.v1.add_to_collection('koi', input['koi'])
      tf.compat.v1.add_to_collection('impression_count_per_day', input['impression_count_per_day'])


      # index [0] (1,1)这种输入 对应eager模式 try_squeeze_dim 会变成(1,) TODO FIXME 可以读取文件设置fixlen_keys 只有这些才会尝试去做try_squeeze
      # 另外tf2 如何infer 应该不同方法?
      if self.need_sigmoid:
        # make pred not 0
        pred = tf.math.sigmoid(self.call(input)) + 1e-5 
      else:
        pred = self.call(input)

      pred = tf.identity(pred, name='pred')

      tf.compat.v1.add_to_collection('logit', self.logit)
      tf.compat.v1.add_to_collection('pred', pred)
      if hasattr(self, 'w'):
        tf.compat.v1.add_to_collection('w', self.w)
      if hasattr(self, 'd'):
        tf.compat.v1.add_to_collection('d', self.d) 

      if hasattr(self, 'w2') and hasattr(self, 'd2'):
        tf.compat.v1.add_to_collection('w2', self.w2)
        tf.compat.v1.add_to_collection('d2', self.d2)

      if hasattr(self, 'y_click') and hasattr(self, 'y_dur'):
        logit_click = tf.identity(self.y_click, name='logit_click')
        tf.compat.v1.add_to_collection('logit_click', logit_click)  
        logit_dur = tf.identity(self.y_dur, name='logit_dur')
        tf.compat.v1.add_to_collection('logit_dur', logit_dur)
        
        pred_click = tf.identity(self.prob_click, name='pred_click')
        tf.compat.v1.add_to_collection('pred_click', pred_click)  
        pred_dur = tf.identity(self.prob_dur, name='pred_dur')
        tf.compat.v1.add_to_collection('pred_dur', pred_dur)
      else:
        pred_click = tf.identity(pred, name='pred_click')
        tf.compat.v1.add_to_collection('pred_click', pred_click)  
        pred_dur = tf.identity(pred, name='pred_dur')
        tf.compat.v1.add_to_collection('pred_dur', pred_dur)

      self.input_feed = input

    FLAGS.is_infer = False


Model = WideDeep
