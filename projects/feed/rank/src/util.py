#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2019-09-09 08:33:48.099948
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

import numpy as np
import time
from datetime import datetime
import re

from tensorflow import keras 
from tensorflow.keras import backend as K
import numpy as np

from projects.feed.rank.src.config import *
from projects.feed.rank.src import evaluate as ev

import melt
import lele
import gezi
logging = gezi.logging

# TODO incl mode for multiple fields
# NOTICE in tf == not work as you expected.. >= > ok
def mask_field(index, value, field, masked_field):
  if FLAGS.mask_mode == 'excl':
    field_mask = tf.cast(K.not_equal(field, masked_field), tf.int64)
  else:
    field_mask = tf.cast(K.equal(field, masked_field), tf.int64) 

  field = field * field_mask
  index = index * field_mask
  field_mask = tf.cast(field_mask, tf.float32)
  value = value * field_mask
  return index, value, field

def get_mask(field, masked_fields):
  mask = None

  if len(masked_fields) == 1:
    if FLAGS.mask_mode == 'excl':
      mask = tf.cast(K.not_equal(field, masked_fields[0]), tf.int64)
    else:
      mask = tf.cast(K.equal(field, masked_field), tf.int64) 
  elif ',' not in FLAGS.masked_fields:
    mask1 = tf.cast(K.greater_equal(field, masked_fields[0]), tf.int64)
    mask2 = tf.cast(K.less(field, masked_fields[-1]), tf.int64)
    mask = mask1 * mask2
    if FLAGS.mask_mode == 'excl':
      mask = 1 - mask

  return mask
  
def mask_fields(index, value, field, masked_fields):
  mask = get_mask(field, masked_fields)
  if mask is None:
    if not FLAGS.mask_use_hash:
      for mfield in masked_fields:
        index, value, field = mask_field(index, value, field, mfield)
      return index, value, field
    
    if not hasattr(mask_fields, 'table'):
      keys = tf.constant(masked_fields, dtype=tf.int64)
      if FLAGS.mask_mode == 'excl': 
        values = tf.constant([0] * len(masked_fields), dtype=tf.int64)
        val = 1 
      else:
        values = tf.constant([1] * len(masked_fields), dtype=tf.int64)
        val = 0

      mask_fields.table = tf.contrib.lookup.HashTable(
      # mask_fields.table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), val)
      if not tf.executing_eagerly():
        with melt.get_session().as_default():
          mask_fields.table.init.run()

    mask = mask_fields.table.lookup(field)

  field = field * mask 
  index = index * mask
  mask = tf.cast(mask, tf.float32)
  value = value * mask 
  return index, value, field

def get_mask_fields(mask_names):
  fields = [] 
  for item in mask_names.split(','):
    if '-' not in item:
      fields.append(np.int64(item))
    else:
      # as using hash not support - anymore
      raise ValueError(mask_names)
      start, end = item.split('-')
      start, end = np.int64(start), np.int64(end)
      fields += range(start, end) 

  return fields

# TODO two places to set col starts here and FLAGS.cb_users
cold_starts = set([931,984,925,926])

def is_cb_user(rea):
  try:
    cb_users = FLAGS.cb_users.split(',')
    if FLAGS.is_infer:
      # for infer inut rea is int not str
      cb_users = [int(x) for x in cb_users]
    rea_w = None
    for i in range(len(cb_users)):
      if i == 0:
        rea_w = tf.cast(tf.math.equal(rea, cb_users[0]), tf.int32)
      else:
        rea_w += tf.cast(tf.math.equal(rea, cb_users[i]), tf.int32)
    return rea_w
  except Exception:
    return False

quality_set = set(['3039', '3057', '3028', '3116', '3065', '3020', \
                   '3008', '3035', '3003', '3019', '3055', '3114', \
                   '3B', '3052', '3147', '3235', '3215', '3031', '3081', \
                   '3212', '3066', '3174', '5051', '3011', '3170', '3361', \
                   '3047', '3148', '3336', '3145', '3221', '3034', '3096', \
                   '3213', '3026', '3070', '3205', 'leapp', '3013', '3062', '3172', \
                   '3400', '3007', '3149', '3337', '3146', '3234', '3214', \
                   '3079', '3211', '3173', '3162', '3340', '1583',
                  ])
def is_quality(x):
  return x in quality_set

def get_product_id(x):
  if FLAGS.is_infer:
    # for infer just pass int 0, 1, 2
    return x
  try:
    return tf.cast(tf.math.equal(x, 'sgsapp'), tf.int32) \
            + tf.cast(tf.math.equal(x, 'newmse'), tf.int32) * 2 \
            + tf.cast(tf.math.equal(x, 'shida'), tf.int32) * 3 \
            - 1
  except Exception:
    return 0
    
# --------- dataset 
def in_days(stamp, days):
  days = set(days)
  return time.localtime(stamp).tm_mday in days 

def get_uid(id):
  return id.decode().split('\t')[0]

def hash_id(ids):
  return np.asarray([gezi.hash_int64(x) for x in ids])
  
# TODO get_finish_ratio for tuwen
def get_finish_ratio(features, max_video_time=None):
  video_time = tf.cast(features['video_time'], tf.float32) 
  duration = tf.cast(features['duration'], tf.float32)
  if max_video_time:
    video_time = tf.math.minimum(video_time, float(max_video_time))
  # in order not to use tf.cond here if video_time unknown then turn it to large video_time 
  # so to make finish_ratio 0.
  video_time = tf.cast(video_time <= 0, tf.float32) * 60000000. + video_time
  finish_ratio = duration / video_time
  finish_ratio = tf.math.minimum(finish_ratio, 1.)
  finish_ratio = tf.math.maximum(finish_ratio, 0.)
  return finish_ratio

def get_time_interval(stamp, time_bins_per_hour=6):
  if not stamp:
    return 0
  x = time.localtime(stamp)
  span = int(60 / time_bins_per_hour)
  return x.tm_hour * time_bins_per_hour + int(x.tm_min / span) + 1

def get_time_intervals(stamps):
  return np.asarray([get_time_interval(x, FLAGS.time_bins_per_hour) + 1 for x in stamps])

def get_weekday(stamp):
  if not stamp:
    return 0
  x = datetime.fromtimestamp(stamp)
  return x.weekday() + 1

# seems not correct for python weekday as datetime.fromtimestamp(1375057496) datetime.datetime(2013, 7, 29, 8, 24, 56)
# this should be Monday by here return 0(Sunday)..
def get_weekdays(stamps):
  return np.asarray([get_weekday(x) + 1 for x in stamps])

def get_distribution_id(x):
  res = tf.numpy_function(hash_id, [x], tf.int64)
  res.set_shape(x.get_shape())
  return res

def get_lookup_array():
  if FLAGS.field_is_hash and FLAGS.field_lookup_container == 'array':
    lookup_arr = [0] * FLAGS.field_hash_size
    lookup_arr[0] = 0
    val = 0
    max_val = 0
    count = 0
    fields = []
    masked_fields = FLAGS.masked_fields.split(',') if FLAGS.masked_fields else None
    mask_mode = FLAGS.mask_mode.replace('_', '-').split('-')[-1] if FLAGS.mask_mode else 'excl'
    for i, line in enumerate(open(FLAGS.field_file_path)):
      l = line.strip().split()
      if len(l) < 2:
        continue
      field_ = l[0]
      if masked_fields:
        if 'regex' in FLAGS.mask_mode:
          if mask_mode == 'excl':
            def _is_ok(x):
              for key in masked_fields:
                if re.search(key, x):
                  return False
              return True
          else:
            def _is_ok(x):
              for key in masked_fields:
                if re.search(key, x):
                  return True
              return False
          if not _is_ok(field_):
            continue
        else:
          if mask_mode == 'excl':
            if field_ in masked_fields:
              continue
          else:
            if field_ not in masked_fields:
              continue
      fields.append(field_)
      fid = int(l[1])
      val = count + 1 if len(l) == 2 else int(l[2])
      if val > max_val:
        max_val = val
      lookup_arr[fid] = val 
      count += 1
    FLAGS.field_dict_size = max_val + 1
    gezi.set_global('fields', fields)
    logging.debug('Final onehot fields is:', ','.join(fields), 'count:', FLAGS.field_dict_size)
    lookup_arr = np.asarray(lookup_arr)
    return lookup_arr
  else:
    return None

def lookup_field(field=None):
  if FLAGS.field_is_hash and FLAGS.field_lookup_container == 'array':
    if not hasattr(lookup_field, 'array'):
      lookup_arr = [0] * FLAGS.field_hash_size
      lookup_arr[0] = 0
      val = 0
      max_val = 0
      count = 0
      fields = []
      masked_fields = FLAGS.masked_fields.split(',') if FLAGS.masked_fields else None
      mask_mode = FLAGS.mask_mode.replace('_', '-').split('-')[-1] if FLAGS.mask_mode else 'excl'
      for i, line in enumerate(open(FLAGS.field_file_path)):
        l = line.strip().split()
        if len(l) < 2:
          continue
        field_ = l[0]
        if masked_fields:
          if 'regex' in FLAGS.mask_mode:
            if mask_mode == 'excl':
              def _is_ok(x):
                for key in masked_fields:
                  if re.search(key, x):
                    return False
                return True
            else:
              def _is_ok(x):
                for key in masked_fields:
                  if re.search(key, x):
                    return True
                return False
            if not _is_ok(field_):
              continue
          else:
            if mask_mode == 'excl':
              if field_ in masked_fields:
                continue
            else:
              if field_ not in masked_fields:
                continue
        fields.append(field_)
        fid = int(l[1])
        val = count + 1 if len(l) == 2 else int(l[2])
        if val > max_val:
          max_val = val
        lookup_arr[fid] = val 
        count += 1
      FLAGS.field_dict_size = max_val + 1
      gezi.set_global('fields', fields)
      logging.debug('Final onehot fields is:', ','.join(fields), 'count:', FLAGS.field_dict_size)
      lookup_arr = np.asarray(lookup_arr)
      if not FLAGS.torch:
        lookup_field.array = melt.layers.LookupArray(lookup_arr, 'field_array')
      else:
        return lele.layers.LookupArray(lookup_arr)

    if field is not None:
      return lookup_field.array(field)
    else:
      return
    
  # NOTICE not tested much, just use above
  if not hasattr(lookup_field, 'table'):
    if FLAGS.field_is_hash:
      # this will be by default, field file is field_name, field_hash, field_class(optional)   
      # like last_7days -323331 0, lasy_3days 323233 0, last_2days 12345 1
      keys = [0]
      values = [0]
      for i, line in enumerate(open(FLAGS.field_file_path)):
        line = line.strip()
        if line:
          l = line.split()
          assert len(l) >= 2
          keys.append(l[1])
          val = i + 1 if len(l) == 2 else int(l[2])
          values.append(val)
    else:
      ## hack for now self increased field input start from 1 and unknown as last
      ## turn to start from 2, unknown as 1, notice use 200 for simple, can just use FLAGS.field_dict_size actually
      keys = tf.constant(list(range(200)), dtype=tf.int64)
      arr = np.asarray(range(200)) + 1
      arr[0] = 0
      values = tf.constant(arr, dtype=tf.int64)
    unknown = 0
    lookup_field.table = tf.contrib.lookup.HashTable(
      tf.lookup.KeyValueTensorInitializer(keys, values), unknown)
    if not tf.executing_eagerly():
      with melt.get_session().as_default():
        lookup_field.table.init.run()

  return lookup_field.table.lookup(field)

# -- bowei
def get_timespan_intervals(impress, pt):
  def get_timespan_interval(a, b):
    if not (a > 0 and b > 0 and a > b):
      return 0
    value = int(np.log2(a - b) * 5) + 1
    if value >200:
      return 200
    return value 
  return np.asarray([get_timespan_interval(impress[i], pt[i]) + 1 for i in range(len(impress))])

#----- for infer
def is_null(x):
  return tf.equal(tf.reduce_sum(x), tf.constant(0, dtype=x.dtype))

def not_null(x):
  return K.not_equal(tf.reduce_sum(x), tf.constant(0, dtype=x.dtype))

def is_infer(input):
  return not_null(input['time_interval'])

def tile(x, n):
  return tf.tile(tf.expand_dims(x, 0), [n, 1])

def tile_embs(embs, n):
  return tf.tile(embs, [n, 1, 1])

def merge_embs(emb, user_index, index):
  ux = emb(tf.expand_dims(user_index, 0))
  ux = tile_embs(ux, melt.get_shape(index, 0))
  x = emb(index)
  return K.concatenate([ux, x], 1)

def get_hash_embedding_type():
  HashEmbedding = getattr(melt.layers, FLAGS.hash_embedding_type)
  HashEmbeddingUD = getattr(melt.layers, FLAGS.hash_embedding_ud_type)

  logging.debug('HashEmbedding:', HashEmbedding, 'combiner:', FLAGS.hash_combiner, 'HashEmbedingUD:', HashEmbeddingUD, 'combiner:', FLAGS.hash_combiner_ud)

  return HashEmbedding, HashEmbeddingUD

# -----train.py related strategy
def get_variables_list():
  variables_list = None
  all_vars = tf.compat.v1.trainable_variables()
  if FLAGS.num_optimizers > 1:
    assert len(FLAGS.optimizers.split(',')) > 1, FLAGS.optimizers
    if FLAGS.vars_split_strategy == 'wide_deep':
      # 0.01
      wide_vars = [x for x in all_vars if x.name.startswith('wide_deep/wide')]
      logging.debug('wide_vars:', wide_vars)
      # 0.001
      deep_vars = [x for x in all_vars if not x.name.startswith('wide_deep/wide')]
      logging.debug('deep_vars:', deep_vars)
      assert wide_vars
      assert deep_vars
      # smaller lr params first
      variables_list = [deep_vars, wide_vars]
    elif FLAGS.vars_split_strategy == 'emb' or FLAGS.vars_split_strategy == 'embeddings':
      # 0.01
      emb_vars = [x for x in all_vars if FLAGS.vars_split_strategy in x.name]
      # 0.001m
      nonemb_vars = [x for x in all_vars if FLAGS.vars_split_strategy not in x.name]
      variables_list = [nonemb_vars, emb_vars]
    elif FLAGS.vars_split_strategy == 'deep_embeddings':
      def _is_embedding(x):
        return 'embeddings' in x.name and '/deep/' in x.name
      emb_vars = [x for x in all_vars if _is_embedding(x)]
      nonemb_vars = [x for x in all_vars if x not in emb_vars]
      variables_list = [nonemb_vars, emb_vars]
    else:
      raise ValueError(FLAGS.vars_split_strategy)
    logging.debug('all vars:', variables_list)
    return variables_list
  else:
    return [all_vars]

def get_eval_fn_and_keys():
  eval_fn = ev.evaluate 
  eval_keys = None
  if FLAGS.eval_rank and not gezi.env_val('EVAL_RANK') == '0':    
    eval_fn = ev.RankEvaluator()
    # with eval_keys for non eager sess.run will be able to get corresponding info like video_time for evaluate.py
    if not FLAGS.compare_online:
      eval_keys = ['mid', 'duration']
    else:
      eval_keys = [
                   'mid', 'docid', 'duration', 'abtestid',
                   'ori_lr_score', 'distribution', 'rea',  
                   'impression_time', 'position', 'video_time', 
                   'show_time', 'lr_score', 'product', 
                   'article_page_time', 'read_completion_rate',
                   'user_active', 
                  ]
  return eval_fn, eval_keys

def prepare_dataset():
  from projects.feed.rank.src.dataset import TextDataset, TFRecordDataset
  Dataset = TextDataset if not 'tfrecord' in FLAGS.train_input else TFRecordDataset

  example = Dataset('train').gen_example(gezi.list_files(FLAGS.train_input.split('|')[0]))
  gezi.set('example', example)
  if example is not None:
    gezi.set_global('embedding_keys', [x[len('index_'):] for x in example if x.startswith('index_')])
  
  if gezi.get_global('embedding_keys') or FLAGS.data_version == 2:
    if os.path.exists(FLAGS.field_file_path):
      keys = [line.strip().split()[0] for line in open(FLAGS.field_file_path)]
      gezi.set_global('embedding_keys', keys)

  return Dataset
