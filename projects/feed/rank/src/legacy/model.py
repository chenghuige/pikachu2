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
import melt

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

from tensorflow import keras
from tensorflow.keras import backend as K

import numpy as np
from projects.feed.rank.src.config import *
from projects.feed.rank.src import util 
import gezi 
logging = gezi.logging

# try:
#   time_module = tf.load_op_library('./ops/time.so')
# except Exception:
#   pass

# output logits!
class Wide(keras.Model):
  def __init__(self):
    super(Wide, self).__init__()
    HashEmbedding = getattr(melt.layers, FLAGS.hash_embedding_type)      
    self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)
    combiner = FLAGS.hash_combiner if FLAGS.hash_combiner != 'concat' else 'sum'
    self.emb = HashEmbedding(FLAGS.wide_feature_dict_size, 1, num_buckets=FLAGS.num_feature_buckets, 
                             embeddings_regularizer=self.regularizer, need_mod=True, 
                             combiner=combiner, name='emb')
    if FLAGS.use_wide_position_emb:
      self.pos_emb = Embedding(FLAGS.num_positions, 1, name='pos_emb')

    if FLAGS.visual_emb:
      melt.visualize_embedding(self.emb, os.path.join(FLAGS.data_dir, 'feature.project'))
      # melt.histogram_summary(self.emb, 'wide.emb')

  # put bias in build so we can track it as WideDeep/Wide/bias
  def build(self, input_shape):
    self.bias = self.add_weight(name='bias',
                                shape=(1,),
                                initializer='zeros',
                                # regularizer=self.regularizer,
                                dtype=tf.float32,
                                trainable=True)

  def call(self, input):
    """outputs is [batch_size, 1]"""
    infer = FLAGS.is_infer
    indexes = input['index']
    fields = input['field']
    values = input['value']  
    if infer:
      uindexes = input['user_index']
      uvalues = input['user_value']
      ufields = input['user_field']

    # if infer:
    #   fields = tf.cond(util.is_null(ufields), lambda: input['field'], lambda: K.concatenate([util.tile(ufields, melt.get_shape(indexes, 0)), input['field']])) 
    #   values = tf.cond(util.is_null(uvalues), lambda: input['value'], lambda: K.concatenate([util.tile(uvalues, melt.get_shape(indexes, 0)), input['value']])) 
    #   pre_indexes = indexes
    #   indexes = tf.cond(util.is_null(uindexes), lambda: pre_indexes, lambda: K.concatenate([util.tile(uindexes, melt.get_shape(indexes, 0)), pre_indexes]))

    # print(len(indexes[0]), len(values[0]), len(fields[0]))
    # print(','.join(map(str, indexes[0].numpy())))
    # print(','.join(map(str, values[0].numpy())))
    # print(','.join(map(str, fields[0].numpy())))
    num_fields = FLAGS.field_dict_size

    if FLAGS.sparse_to_dense:
      with melt.device(FLAGS.hack_device):
        if infer:
          ## TODO FIXME can not dump
          # File "/home/gezi/env/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/graph_util_impl.py", line 297, in convert_variables_to_constants
          # source_op_name = get_input_name(node)
          # File "/home/gezi/env/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/graph_util_impl.py", line 254, in get_input_name
          # raise ValueError("Tensor name '{0}' is invalid.".format(node.input[0]))
          # ValueError: Tensor name 'wide/cond_3/emb/embedding_lookup/Switch:1' is invalid.
          
          # x = tf.cond(util.is_null(uindexes), lambda: self.emb(indexes), lambda: util.merge_embs(self.emb, uindexes, pre_indexes))
          x = self.emb(indexes)
        else:
          x = self.emb(indexes)
 
      x = K.squeeze(x, -1)
      
      if not FLAGS.use_wide_val:
        values = input['binary_value']

      x = x * values
      
      if FLAGS.use_fm_first_order:
        x = tf.expand_dims(x, -1)
        x = melt.unsorted_segment_embs(x, fields, num_fields, combiner=FLAGS.pooling)
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

    if not FLAGS.use_fm_first_order:
      x = x + self.bias
    return x  

# TODO try bottom mlp as drlm
class Deep(keras.Model):
  def __init__(self):
    super(Deep, self).__init__()
    # # do not need two many deep embdding, only need some or cat not cross TODO
    # # STILL OOM FIXME...
    # if FLAGS.hidden_size > 50:
    #   print('---------------put emb on cpu')
    #   with tf.device('/cpu:0'):
    #     self.emb = keras.layers.Embedding(FLAGS.feature_dict_size + 1, FLAGS.hidden_size)
    # else:
    self.regularizer = keras.regularizers.l1_l2(l2=FLAGS.l2_reg)

    Embedding = melt.layers.Embedding

    assert FLAGS.sparse_to_dense
    
    HashEmbedding, HashEmbeddingUD = util.get_hash_embedding_type()

    kwargs = dict(num_buckets=FLAGS.num_feature_buckets, combiner=FLAGS.hash_combiner, 
                  embeddings_regularizer=self.regularizer, append_weight=FLAGS.hash_append_weight)
    self.emb = HashEmbedding(FLAGS.feature_dict_size, FLAGS.hidden_size, name='emb', **kwargs)
  
    if FLAGS.use_user_emb:
      self.user_emb = HashEmbeddingUD(FLAGS.feature_dict_size, FLAGS.hidden_size, name='user_emb', **kwargs)
    if FLAGS.use_doc_emb:
      self.doc_emb = HashEmbeddingUD(FLAGS.feature_dict_size, FLAGS.hidden_size, name='doc_emb', **kwargs)
      
    if FLAGS.use_history_emb:
      # TODO
      # self.kw_emb = Embedding(1000000, FLAGS.hidden_size, name='kw_emb')
      self.kw_emb = Embedding(FLAGS.keyword_dict_size, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='kw_emb')
      # 10000
      self.topic_emb = Embedding(FLAGS.topic_dict_size, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='topic_emb')

      # if FLAGS.history_attention:
      #   self.history_att = melt.layers.DotAttention(FLAGS.hidden_size)

    if FLAGS.use_time_emb:
      self.time_emb = Embedding(500, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='time_emb')
      self.weekday_emb = Embedding(10, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='weekday_emb')

    if FLAGS.use_timespan_emb:
      self.timespan_emb = Embedding(300, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='timespan_emb')

    if FLAGS.use_deep_position_emb:
      self.pos_emb = Embedding(FLAGS.num_positions, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='pos_emb')

    if FLAGS.use_product_emb:
      self.product_emb = Embedding(10, FLAGS.hidden_size, embeddings_regularizer=self.regularizer, name='product_emb')
    # # Not work..
    # # num_shards = FLAGS.num_gpus
    # num_shards = melt.num_gpus2()
    # gezi.sprint(num_shards)
    # if num_shards > 1:
    #   Embedding = melt.layers.Embedding
    #   # TODO tf2.0
    #   with tf.device(tf.train.replica_device_setter(ps_tasks=num_shards)):
    #     self.emb = Embedding(FLAGS.feature_dict_size, FLAGS.hidden_size, num_shards=num_shards, name='emb')  

    if FLAGS.visual_emb:
      melt.visualize_embedding(self.emb, os.path.join(FLAGS.data_dir, 'feature.project'))
      # melt.histogram_summary(self.emb, 'deep.emb')
    self.emb_dim = FLAGS.hidden_size
    if FLAGS.field_emb:
      embeddings_initializer = 'uniform' if not FLAGS.disable_field_emb else 'zeros'
      self.field_emb = Embedding(FLAGS.field_dict_size, FLAGS.hidden_size, embeddings_initializer=embeddings_initializer, name='field_emb')
      if FLAGS.visual_emb:
        melt.visualize_embedding(self.field_emb, os.path.join(FLAGS.data_dir, 'field.project'))
        # melt.histogram_summary(self.field_emb, 'deep.field_emb')  # TODO how to do histogram for keras layers without using model.fit train and tensorboard callback
      self.emb_dim += FLAGS.hidden_size

    if FLAGS.use_fm:
      self.bias_emb = Embedding(FLAGS.feature_dict_size, 1, name='bias_emb')
    
    self.emb_activation = None
    if FLAGS.emb_activation:
      self.emb_activation = keras.layers.Activation(FLAGS.emb_activation)

    if not FLAGS.mlp_dims:
      self.mlp = None
    else:
      dims = [int(x) for x in FLAGS.mlp_dims.split(',')]
      activation = FLAGS.dense_activation if not FLAGS.mlp_norm else None
      drop_rate = FLAGS.mlp_drop if not FLAGS.mlp_norm else None
      self.mlp = melt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
      self.bottom_mlp = melt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
      if FLAGS.multi_obj_type == 'shared_bottom':
        self.mlp2 = melt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
        self.bottom_mlp2 = melt.layers.MLP(dims, activation=activation,
            drop_rate=drop_rate)
      if FLAGS.use_task_mlp:
        self.task_mlp = melt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)
        self.task_mlp2 = melt.layers.MLP(dims, activation=activation,
          drop_rate=drop_rate)

      # bad result
      if FLAGS.mlp_norm:
        self.batch_norm = tf.keras.layers.BatchNormalization()  

    act = FLAGS.dense_activation if FLAGS.deep_final_act else None    
    if FLAGS.num_experts:
      self.mmoe = melt.layers.MMoE(FLAGS.hidden_size, num_experts=FLAGS.num_experts, num_tasks=2)

    # NOTICE as we always * values then actually do not need pooling if sum or mean but for max still need
    if FLAGS.pooling != 'allsum':
      self.pooling = melt.layers.Pooling(FLAGS.pooling)
    self.sum_pooling = melt.layers.Pooling('sum')

    if FLAGS.use_fm or FLAGS.use_slim_fm:
      self.fm_pooling = melt.layers.Pooling('fm')
      
    if FLAGS.use_history_emb:
      self.hpooling = melt.layers.Pooling(FLAGS.hpooling)

    if FLAGS.field_pooling:
      assert FLAGS.field_concat
      self.field_pooling = melt.layers.Pooling(FLAGS.field_pooling, att_hidden=50)

    if FLAGS.emb_drop:
      self.dropout = keras.layers.Dropout(FLAGS.emb_drop)

    if FLAGS.deep_out_dim == 1:
      self.dense = keras.layers.Dense(1)
      if FLAGS.multi_obj_type:
        self.dense2 = keras.layers.Dense(1)

  def build(self, input_shape):
    if FLAGS.emb_activation:
      self.bias = K.variable(value=[0.], name='bias')

  def call(self, input):
    """outputs is [batch_size, 1]"""
    infer = FLAGS.is_infer
    indexes = input['index']
    fields = input['field']
    values = input['value']  
    if infer:
      uindexes = input['user_index']
      uvalues = input['user_value']
      ufields = input['user_field']

    # if infer:
    #   fields = tf.cond(util.is_null(ufields), lambda: input['field'], lambda: K.concatenate([util.tile(ufields, melt.get_shape(indexes, 0)), input['field']])) 
    #   values = tf.cond(util.is_null(uvalues), lambda: input['value'], lambda: K.concatenate([util.tile(uvalues, melt.get_shape(indexes, 0)), input['value']])) 
    #   pre_indexes = indexes
    #   indexes = tf.cond(util.is_null(uindexes), lambda: pre_indexes, lambda: K.concatenate([util.tile(uindexes, melt.get_shape(indexes, 0)), pre_indexes]))
      
    if FLAGS.sparse_to_dense:
      x_len = melt.length(indexes) 
   
    num_fields = FLAGS.field_dict_size 
    dense_feats = []
    other_embs = []

    if FLAGS.use_product_emb:
      x_product = self.product_emb(util.get_product_id(input['product']))
      other_embs += [x_product]

    if FLAGS.use_user_emb:
      # this will output [bs, 1, hidden_size] 
      with melt.device(FLAGS.hack_device):
        x_user = self.user_emb(input['uid'])
      x_user = tf.squeeze(x_user, 1)
      other_embs += [x_user]

    if FLAGS.use_doc_emb:
      with melt.device(FLAGS.hack_device):
        x_doc = self.doc_emb(input['did']) 
      x_doc = tf.squeeze(x_doc, 1)
      other_embs += [x_doc]
      
    if FLAGS.use_history_emb:
      histories = input['history']
      # topics = input['topic']
      # keywords = input['keyword']
      
      # doc_keywords = input['doc_keyword']
      # doc_topic = input['doc_topic']
      
      if not FLAGS.history_attention:
        x_hist = self.hpooling(self.doc_emb(histories), melt.length(histories))
      else:
        # A = self.history_att(self.doc_emb(histories), tf.expand_dims(x_doc, 1), mask=tf.cast(input['history'], tf.bool))
        with melt.device(FLAGS.hack_device):
          history = self.doc_emb(histories)
        query = tf.transpose(tf.expand_dims(x_doc, 1), [0, 2, 1])
        # print(history, query)
        weight = tf.matmul(history, query)
        outs = history * weight
        x_hist = self.hpooling(outs)

      # B = self.hpooling(self.topic_emb(topics), melt.length(topics))
      # C = self.hpooling(self.kw_emb(keywords), melt.length(keywords))
      # D = self.hpooling(self.topic_emb(doc_topic), melt.length(doc_topic))
      # E = self.hpooling(self.kw_emb(doc_keywords), melt.length(doc_keywords))

      other_embs += [x_hist]

    if FLAGS.use_time_emb:
      if FLAGS.use_time_so:
        get_time_intervals = time_module.time
      else:
        def get_time_intervals(x):
          res = tf.numpy_function(util.get_time_intervals, [x], tf.int64)
          res.set_shape(x.get_shape())
          return res

      time_interval = input['time_interval']
      if FLAGS.time_smoothing:
        x_time = self.time_emb(time_interval)
        num_bins = FLAGS.time_bins_per_hour * 24
        tmask = tf.cast(time_interval > 1, tf.int64)
        tbase = time_interval * (1 - tmask)
        time_pre = (time_interval - 2 -1 * FLAGS.time_bins_per_hour) % num_bins + 2 
        time_pre = tbase + time_pre * tmask
        time_pre2 = (time_interval - 2 -2 * FLAGS.time_bins_per_hour) % num_bins + 2
        time_pre2 = tbase + time_pre2 * tmask
        time_after = (time_interval - 2 + 1 * FLAGS.time_bins_per_hour) % num_bins + 2
        time_after = tbase + time_after * tmask
        time_after2 = (time_interval - 2 + 2 * FLAGS.time_bins_per_hour) % num_bins + 2
        time_after2 = tbase + time_after2 * tmask
        x_time_pre = self.time_emb(time_pre)
        x_time_pre2 = self.time_emb(time_pre2)
        x_time_after = self.time_emb(time_after)
        x_time_after2 = self.time_emb(time_after2)
        x_time = (0.4 * x_time + 0.2 * x_time_pre + 0.1 * x_time_pre2 + 0.2 * x_time_after + 0.1 * x_time_after2) / 5.
      # print('x_time2', x_time)
      elif FLAGS.time_bins_per_day:
        num_bins = FLAGS.time_bins_per_hour * 24
        num_large_bins = FLAGS.time_bins_per_day
        intervals_per_large_bin = tf.cast(num_bins / num_large_bins, tf.int64)
        tmask = tf.cast(time_interval > 1, tf.int64)
        tbase = time_interval * (1 - tmask)
        time_interval_large = tf.cast(((time_interval - 2 - FLAGS.time_bin_shift_hours * FLAGS.time_bins_per_hour) % num_bins)/ intervals_per_large_bin, tf.int64) + 2
        time_interval_large = tbase + time_interval_large * tmask
        x_time = self.time_emb(time_interval_large)
      else:
        x_time = self.time_emb(time_interval)

      time_weekday = input['time_weekday'] 
      x_weekday = self.weekday_emb(time_weekday)
      other_embs += [x_time, x_weekday]      

      if FLAGS.use_dense_feats:
        # TODO remove dense feats of time as 23 and 00... 
        s_time = tf.cast(time_interval, tf.float32) / (24 * FLAGS.time_bins_per_hour + 10.)
        s_time = tf.zeros_like(time_interval)
        s_time = melt.youtube_scalar_features(s_time)

        s_weekday = tf.cast(time_weekday, tf.float32) / 10.
        s_weekday = melt.youtube_scalar_features(s_weekday)

        dense_feats = [s_time, s_weekday]

    if FLAGS.use_timespan_emb:
      if FLAGS.use_time_so:
        get_timespan_intervals = time_module.timespan
      else:
        def get_timespan_intervals(x, y): 
          res = tf.numpy_function(util.get_timespan_intervals, [x, y], tf.int64)
          res.set_shape(x.get_shape())
          return res

      timespan_interval = input['timespan_interval']
      x_timespan = self.timespan_emb(timespan_interval)
      other_embs += [x_timespan]

      if FLAGS.use_dense_feats:
        s_timespan = tf.cast(timespan_interval, tf.float32) / 200. 
        s_timespan = melt.youtube_scalar_features(s_timespan)
        dense_feats += [s_timespan]

        s_timespan2 = input['impression_time'] - input['article_page_time']
        max_delta = 3000000
        s_timespan2 = tf.math.minimum(s_timespan2, max_delta)
        # s_timespan2 = tf.math.maximum(s_timespan2, -10)
        s_timespan2 = tf.math.maximum(s_timespan2, 0)
        s_timespan2 = tf.cast(s_timespan2, tf.float32) / float(max_delta)
        s_timespan2 = melt.youtube_scalar_features(s_timespan2)
        dense_feats += [s_timespan2]
    
    # just use sparse_to_dense 
    if FLAGS.use_onehot_emb:
      if fields:
        if not FLAGS.sparse_to_dense:    
          assert FLAGS.pooling == 'sum' or FLAGS.pooling == 'mean'
          assert not FLAGS.field_concat, "TODO.."
          values_ = values if FLAGS.use_deep_val else None
          with melt.device(FLAGS.hack_device):
            x = tf.nn.embedding_lookup_sparse(params=self.emb(None), sp_ids=indexes, sp_weights=values_, combiner=FLAGS.pooling)
          if FLAGS.field_emb:
            x = K.concatenate([x, tf.nn.embedding_lookup_sparse(params=self.field_emb(None), sp_ids=fields, sp_weights=None, combiner=FLAGS.pooling)], axis=-1) 
          return x
        else:
          with melt.device(FLAGS.hack_device):
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

          if not FLAGS.use_deep_val:
            values = input['binary_value']      
          values = K.expand_dims(values, -1)
          x = x * values

          if FLAGS.use_fm:
            # Too much calcuations here not to use
            assert not FLAGS.use_slim_fm
            if FLAGS.use_other_embs_fm and other_embs:
              x_other = tf.stack(other_embs, 1)
              if FLAGS.field_emb:
                num_others = len(other_embs)
                # TODO FIXME should be num_fields + num other embs total
                other_fields = tf.ones_like(indexes[:, :num_others]) + tf.constant(list(range(len(other_embs))), dtype=tf.int64)
                x_other_field = self.field_emb(other_fields)
                x_other = K.concatenate([x_other, x_other_field])
              x_all = K.concatenate([x, x_other], 1)
              all_len = x_len + len(other_embs) 
            else:
              x_all = x
              all_len = x_len

            fm = self.second_order_fm(x_all, all_len)

          num_fields_ = 1
          if FLAGS.field_concat:
            num_fields_ = num_fields - 1
            # x = tf.math.unsorted_segment_sum(x, fields, num_fields)  
            # like [batch_size * max_dim, hidden_dim] ->   [batch_size * num_segs, hidden_dim] -> [batch_size, num_segs, hidden_dim]
            # TODO mask and filter out zero embeddings 
            x = melt.unsorted_segment_embs(x, fields, num_fields, combiner=FLAGS.pooling)
            # TODO do not need reshape ? change unsorted ..
            x = x[:, 1:, :]
            xs = x
            if not FLAGS.field_pooling:
              # like [batch_size, num_segs * hidden_dim], ignore zero index
              x = K.reshape(x, [-1, num_fields_ * self.emb_dim])
            else:
              x = self.field_pooling(x)
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
      else:
        pass
    if FLAGS.use_slim_fm:
      assert not FLAGS.use_fm
      if other_embs:
        x_other = tf.stack(other_embs, 1)
      if FLAGS.use_onehot_emb:
        if other_embs:
          x_all = K.concatenate([xs, x_other], 1)
        else:
          x_all = xs
      else:
        x_all = x_other
        num_fields_ = 0
        assert len(other_embs)
      # all_len = tf.zeros_like(x_len) + len(other_embs) + num_fields_
        
      fm = self.fm_pooling(x_all)

    # TODO this is concat pooling
    if FLAGS.use_onehot_emb:
      if other_embs:
        all_embs = [x] + other_embs
        x = K.concatenate(all_embs)
    else:
      assert len(other_embs)
      x = K.concatenate(other_embs)
    
    if self.emb_activation:
      x = self.emb_activation(x + self.bias)
            
    if FLAGS.id_feature_only:
      if FLAGS.user_emb_only:
        x = x_user
      elif FLAGS.doc_emb_only:
        x = x_doc 
      else:
        x = K.concatenate([x_user, x_doc])
      
    if FLAGS.emb_drop:
      x = self.dropout(x)
      
    if FLAGS.use_fm or FLAGS.use_slim_fm and FLAGS.fm_before_mlp:
      x = K.concatenate([fm, x])

    if FLAGS.num_experts:
      xs = self.mmoe(x) 
      x = xs[0]
      x2 = xs[1]
    else:
      x2 = x 
      
    # TODO better support dense feats with feature intersection
    # if dense_feats:
    #   dense_feats =K.concatenate(dense_feats)
    #   dx = self.bottom_mlp(dense_feats)
    #   x = K.concatenate([x, dx])
    #   # if FLAGS.multi_obj_type:
    #   #   dx2 = self.bottom_mlp2(dense_feats)
    #   #   x2 = K.concatenate([x2, dx2])
      
    self.before_mlp = x
  
    if self.mlp:
      # need this for fm .. may be should shring fields size by count dict not hash to reduce input dim to mlp TODO
      # This makes training much slower 8 -> 4 it/s
      device = None if (not FLAGS.field_concat or FLAGS.field_dict_size < 1000) else FLAGS.hack_device
      with melt.device(device):
        x = self.mlp(x)
        if FLAGS.multi_obj_type:
          # if share mlp not need to do mlp2 but for old models compat here
          if (not FLAGS.multi_obj_share_mlp) or FLAGS.compat_old_model:
            x2 = self.mlp2(x2)
          else:
            x2 = x

    self.after_mlp = x

    if FLAGS.mlp_norm:
      x = self.batch_norm(x)
  
    if FLAGS.multi_obj_type and FLAGS.use_task_mlp:
      assert FLAGS.multi_obj_share_mlp
      x2 = self.task_mlp2(x)
      x = self.task_mlp(x)

    if FLAGS.use_fm or FLAGS.use_slim_fm and (not FLAGS.fm_before_mlp):
      x = K.concatenate([fm, x])
      x2 = K.concatenate([fm, x2])

    if FLAGS.use_deep_position_emb:
      position = input['position']
      x_pos = self.pos_emb(position)
      def merge_pos(x, x_pos):
        if FLAGS.position_combiner == 'concat':
          x = K.concatenate([x, x_pos])
        elif FLAGS.position_combiner == 'add':
          x = x + x_pos
        else:
          raise ValueError('Unsuported position_combiner %s' % FLAGS.position_combiner)
      x = merge_pos(x, x_pos)
      if FLAGS.multi_obj_type:
        x2 = merge_pos(x2, x_pos)

    if FLAGS.deep_out_dim == 1:
      x = self.dense(x)
      if FLAGS.multi_obj_type:
        x2 = self.dense2(x2)

    if FLAGS.multi_obj_type:
      self.x2 = x2

    return x

class WideDeep(keras.Model):   
  def __init__(self):
    super(WideDeep, self).__init__()

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

  # # https://towardsdatascience.com/everything-you-need-to-know-about-tensorflow-2-0-b0856960c074
  # # You need to override this function if you want to use the subclassed model
  # # as part of a functional-style model.
  # # Otherwise, this method is optional.
  # def compute_output_shape(self, input_shape):
  #   shape = tf.TensorShape(input_shape).as_list()
  #   shape[-1] = 1
  #   return tf.TensorShape(shape)

  def adjust_input(self, input):
    if FLAGS.need_field_lookup:
      # NOTICE you can only look up onece as now you need to first %  then lookup
      if input['field']:
        input['field'] = util.lookup_field(input['field'])

    # TODO HACK now for input val problem we turn 0 value to 1.
    # NOTICE here padding value 0 is also turn to 1 if not mask by index
    if not FLAGS.ignore_zero_value_feat:
      if not isinstance(input['index'], dict):
        mask = tf.cast(K.equal(input['index'], 0), tf.float32)
        input['value'] += (tf.cast(K.equal(input['value'], 0.), tf.float32) - mask)
      else:
        for key in input['index']:
          mask = tf.cast(K.equal(input['index'][key], 0), tf.float32)
          input['value'][key] += (tf.cast(K.equal(input['value'][key], 0.), tf.float32) - mask)
    
    # mask some field's according value to 0. so as to mask those fields
    if FLAGS.masked_fields:
      assert input['field']
      masks = util.get_mask_fields(FLAGS.masked_fields)
      input['index'], input['value'], input['field'] = util.mask_fields(input['index'], input['value'], input['field'], masks)

    if (not FLAGS.use_wide_val) or (not FLAGS.use_deep_val):
      if not isinstance(input['value'], dict):
        input['binary_value'] = tf.cast(K.not_equal(input['value'], 0), tf.float32)    
      else:
        input['binary_value'] = {}
        for key in input['value']:
          input['binary_value'][key] = tf.cast(K.not_equal(input['value'][key], 0), tf.float32)

    tf.compat.v1.add_to_collection('index', input['index'])
    tf.compat.v1.add_to_collection('value', input['value'])
    tf.compat.v1.add_to_collection('field', input['field'])

  # @tf.function  
  def call(self, input):
    """outputs is [batch_size, 1]"""
    self.adjust_input(input)

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
        
    if melt.get_shape(x, 1) > 1:
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
      if melt.get_shape(x2, 1) > 1:
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
        product = tf.expand_dims(product, -1)

        click_powers = list(map(float, FLAGS.click_power.split(',')))
        dur_powers = list(map(float, FLAGS.dur_power.split(',')))

        click_power = tf.nn.embedding_lookup(tf.constant(click_powers), product)
        dur_power = tf.nn.embedding_lookup(tf.constant(dur_powers), product)

        cb_click_powers = list(map(float, FLAGS.cb_click_power.split(',')))
        cb_dur_powers = list(map(float, FLAGS.cb_dur_power.split(',')))
        cb_click_power = tf.nn.embedding_lookup(tf.constant(cb_click_powers), product)
        cb_dur_power = tf.nn.embedding_lookup(tf.constant(cb_dur_powers), product)     

        cb = tf.cast(util.is_cb_user(input['rea']), tf.float32)
        cb = tf.expand_dims(cb, -1)

        # Notice bad name here.. actually should use name as exponent not power
        click_power = click_power * (1.0 - cb) + cb_click_power * cb
        dur_power = dur_power * (1.0 - cb) + cb_dur_power * cb
        
        dur_need_sigmoid = True
        if 'jump' in FLAGS.multi_obj_duration_loss:
          if not 'cross_entropy' in FLAGS.jump_loss:
            dur_need_sigmoid = False
        elif not 'cross_entropy' in FLAGS.multi_obj_duration_loss:
          dur_need_sigmoid =False  
        
        self.prob_dur = tf.math.sigmoid(self.y_dur) if dur_need_sigmoid else self.y_dur
        if 'merge' in FLAGS.multi_obj_duration_loss and K.learning_phase():
          click_power = 1.
          dur_power = 1.

        if FLAGS.dynamic_multi_obj_weight:
          click_power *= tf.gather(self.deep.multi_weights, [0], axis=1)
          dur_power *= tf.gather(self.deep.multi_weights, [1], axis=1)
        
        # TODO numeric stable ?
        y = (self.prob_click ** click_power) * (self.prob_dur ** dur_power)
        if FLAGS.finish_loss:
          y *= self.prob_finish ** FLAGS.finish_power
        y = y ** 0.5
        self.prob = y
        if 'dense' in FLAGS.multi_obj_duration_loss:
          y = self.dense_merge(K.concatenate([self.y_click, self.y_dur], 1))
        else:
          # always here
          y = melt.prob2logit(y)

    # output shape is (batch_size, 1)
    self.pred = y
    self.logit = y
    self.need_sigmoid = True
    
    # TODO valid loss not ok ? though for evaluate.py output is ok
    if K.learning_phase() == 0:
      if self.prob is None:
        pred = tf.math.sigmoid(y)
        self.prob = pred
      else:
        pred = self.prob
      self.need_sigmoid = False
      if FLAGS.debug_infer:
        pred = tf.cast(tf.cast(pred * 10000., tf.int32), tf.float32) / 10000. + 0.0000111 * FLAGS.model_mark
      return pred

    return y

  def init_predict(self):
    # TODO use decorator @is_infer
    K.set_learning_phase(0)
    FLAGS.is_infer = True
    # # TODO  infer mode can not dump pb cond embedding lookup name invalid

    # TODO sparse 
    if not FLAGS.sparse_to_dense:
      logging.warning('Unsuported for sparse input now')
      return

    index_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'index_feed')
    tf.compat.v1.add_to_collection('index_feed', index_feed)

    value_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0.]], dtype=tf.float32), [None, None], 'value_feed')
    tf.compat.v1.add_to_collection('value_feed', value_feed)

    field_feed = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'field_feed')
    tf.compat.v1.add_to_collection('field_feed', field_feed)

    input = dict(index=index_feed, value=value_feed, field=field_feed)

    dummy = tf.zeros_like(index_feed[:, 0])
    input['position'] = dummy

    dummy = tf.expand_dims(dummy, 1)

    input['user_index'] = tf.compat.v1.placeholder_with_default(tf.constant([0], dtype=tf.int64), [None], 'user_index_feed')
    tf.compat.v1.add_to_collection('user_index_feed', input['user_index'])

    input['user_value'] = tf.compat.v1.placeholder_with_default(tf.constant([0.], dtype=tf.float32), [None], 'user_value_feed')
    tf.compat.v1.add_to_collection('user_value_feed', input['user_value'])

    input['user_field'] = tf.compat.v1.placeholder_with_default(tf.constant([0], dtype=tf.int64), [None], 'user_field_feed')
    tf.compat.v1.add_to_collection('user_field_feed', input['user_field'])

    # if FLAGS.use_user_emb:
    input['uid'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'uid_feed')
    tf.compat.v1.add_to_collection('uid_feed', input['uid'])
    input['uid'] += dummy

    # if FLAGS.use_doc_emb:
    input['did'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'did_feed')
    tf.compat.v1.add_to_collection('did_feed', input['did'])
    input['did'] += dummy

    input['history'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'doc_idx_feed')
    tf.compat.v1.add_to_collection('doc_idx_feed', input['history'])
    input['history'] += dummy

    input['keyword'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'kw_idx_feed')
    tf.compat.v1.add_to_collection('kw_idx_feed', input['keyword'])
    input['keyword'] += dummy

    input['topic'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None], 'topic_idx_feed')
    tf.compat.v1.add_to_collection('topic_idx_feed', input['topic'])
    input['topic'] += dummy

    input['doc_keyword'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, None],'doc_kw_idx_feed')
    tf.compat.v1.add_to_collection('doc_kw_idx_feed', input['doc_keyword'])
    input['doc_keyword'] += dummy

    input['doc_topic'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'doc_topic_idx_feed')
    tf.compat.v1.add_to_collection('doc_topic_idx_feed', input['doc_topic'])
    input['doc_topic'] += dummy

    input['impression_time'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'time_feed')
    tf.compat.v1.add_to_collection('time_feed', input['impression_time'])
    input['impression_time'] += dummy
    input['impression_time'] = tf.squeeze(input['impression_time'], 1)

    input['article_page_time'] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=tf.int64), [None, 1], 'ptime_feed')
    tf.compat.v1.add_to_collection('ptime_feed', input['article_page_time'])
    input['article_page_time'] += dummy
    input['article_page_time'] = tf.squeeze(input['article_page_time'], 1)

    def _add(name, dtype=tf.int64):
      input[name] = tf.compat.v1.placeholder_with_default(tf.constant([[0]], dtype=dtype), [None, 1], '%s_feed' % name)
      tf.compat.v1.add_to_collection('%s_feed' % name, input[name])
      input[name] += dummy
      input[name] = tf.squeeze(input[name], 1)
    
    def _adds(names, dtype=tf.int64):
      for name in names:
        _add(name, dtype)

    # Notice we use time_interval to judge online infer or not when online infer time interval is one of the input but for offline using infer.py not
    names = [
      'product',
      'time_interval', 'time_weekday', 'timespan_interval', 'device_info', 'rea', 
      'network', 'user_activity', 'impression_count', 'impression_count_per_day',
      'koi'  # TODO FIXME.. strange why add koi or anything here will infer output shape always be [1,1] ??
    ]

    _adds(names, tf.int64)

    # TODO just for debug
    tf.compat.v1.add_to_collection('dummy', dummy)
    tf.compat.v1.add_to_collection('uid', input['uid'])
    tf.compat.v1.add_to_collection('product', input['product'])
    tf.compat.v1.add_to_collection('time_interval', input['time_interval'])
    tf.compat.v1.add_to_collection('koi', input['koi'])
    tf.compat.v1.add_to_collection('impression_count_per_day', input['impression_count_per_day'])

    pred = self.call(input)
    if self.need_sigmoid:
      # make pred not 0
      pred = tf.math.sigmoid(self.call(input)) + 1e-5 

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

    FLAGS.is_infer = False

