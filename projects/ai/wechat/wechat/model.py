#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:25.245765
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import ge
from re import A

import sys 
import os
import numpy as np
import copy
from icecream import ic

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import gezi
from gezi import logging
import melt as mt
from wechat.config import *
from wechat import util
from wechat.util import *
from wechat.encoder import *

class Model(mt.Model):
  def __init__(self):
    super(Model, self).__init__() 
    self.input_ = None

    Embedding = get_embedding
    emb_dim = FLAGS.emb_dim # 当前所有emb 都按照128 pretrain
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
    # 这个不能用 否则eva 虚高 test没有is_first==False
    self.is_first_emb = Embedding('is_first', 2)

    self.position_emb = Embedding('position', 100)
    self.span_emb = Embedding('span', 20)

    util.init_lookups()

    if 'docs' in FLAGS.his_id_feats:
      self.docs_emb = get_docs_emb()

    self.feed_emb = Embedding('feed')
    if FLAGS.doc2_emb:
      self.doc2_emb = Embedding('doc2')
    if FLAGS.desc_vec_emb:
      self.desc_vec_emb = Embedding('desc_vec')
    if FLAGS.ocr_vec_emb:
      self.ocr_vec_emb = Embedding('ocr_vec')
    if FLAGS.asr_vec_emb:
      self.asr_vec_emb = Embedding('asr_vec')

    if FLAGS.feed_mlp:
      self.feed_dense = mt.layers.MLP([256, FLAGS.emb_dim], activate_last=False, name='feed_dense')
    elif FLAGS.feed_project:
      self.feed_dense = mt.layers.Project(FLAGS.emb_dim, name='feed_dense')
    else:
      self.feed_dense = keras.layers.Dense(FLAGS.emb_dim, name='feed_dense') 
      
    self.doc_dense = keras.layers.Dense(FLAGS.emb_dim, name='doc_dense')
    self.user_dense = keras.layers.Dense(FLAGS.emb_dim, name='user_dense')
    self.docs_dense = keras.layers.Dense(FLAGS.emb_dim, name='docs_dense')

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
      'fresh2': self.fresh_emb,
    }
    for key in FLAGS.doc_keys:
      try:
        self.embs[key] = getattr(self, f'{key}_emb')
      except Exception:
        pass

    self.tag_emb = Embedding('tag')
    self.key_emb = Embedding('key')
    self.char_emb = Embedding('char')
    self.word_emb = Embedding('word')

    self.action_emb = Embedding('action', 100)

    self.key_dense = keras.layers.Dense(FLAGS.emb_dim, name='key_dense')
    self.word_dense = keras.layers.Dense(FLAGS.emb_dim, name='word_dense')
    self.char_dense = keras.layers.Dense(FLAGS.emb_dim, name='char_dense')

    if FLAGS.share_tag_encoder:
      self.tag_encoder = Encoder(self.tag_emb, pooling=FLAGS.encoder_pooling, emb_dim=FLAGS.emb_dim, num_lookups=MAX_TAGS, name='tag_encoder')
    else:
      self.manual_tags_encoder = Encoder(self.tag_emb, pooling=FLAGS.encoder_pooling, emb_dim=FLAGS.emb_dim, name='manual_tags_encoder')
      self.machine_tags_encoder = Encoder(self.tag_emb, pooling=FLAGS.encoder_pooling, emb_dim=FLAGS.emb_dim, num_lookups=MAX_TAGS, name='machine_tags_encoder')
   
    if FLAGS.share_key_encoder:
      self.key_encoder = Encoder(self.key_emb, pooling=FLAGS.encoder_pooling, name='key_encoder')
    else:
      self.manual_keys_encoder = Encoder(self.key_emb, pooling=FLAGS.encoder_pooling, name='manual_keys_encoder')
      self.machine_keys_encoder = Encoder(self.key_emb, pooling=FLAGS.encoder_pooling, name='machine_keys_encoder')

    if FLAGS.share_text_encoder:
      self.word_encoder = Encoder(self.word_emb, pooling=FLAGS.encoder_pooling, name='word_encoder')
      self.char_encoder = Encoder(self.char_emb, pooling=FLAGS.encoder_pooling, name='char_encoder')
    else:
      self.desc_encoder = Encoder(self.word_emb, pooling=FLAGS.encoder_pooling, name='desc_encoder')
      self.desc_char_encoder = Encoder(self.char_emb, pooling=FLAGS.encoder_pooling, name='desc_char_encoder')

      self.ocr_encoder = Encoder(self.word_emb, pooling=FLAGS.encoder_pooling, name='ocr_encoder')
      self.ocr_char_encoder = Encoder(self.char_emb, pooling=FLAGS.encoder_pooling, name='ocr_char_encoder')

      self.asr_encoder = Encoder(self.word_emb, pooling=FLAGS.encoder_pooling, name='asr_encoder')
      self.asr_char_encoder = Encoder(self.char_emb, pooling=FLAGS.encoder_pooling, name='asr_char_encoder')  

    self.pooling = mt.layers.Pooling(FLAGS.pooling, name='ori')
    self.pooling2 = mt.layers.Pooling(FLAGS.pooling2, name='normed')
    self.his_pooling = mt.layers.Pooling(FLAGS.his_pooling, name='his')
    self.his_pooling2 = mt.layers.Pooling(FLAGS.his_pooling, name='his2')

    keys = ['doc'] + FLAGS.doc_keys 
    user_keys = ['user']
    din_keys = FLAGS.din_keys
    self.his_poolings = {}
    for key in keys: # doc feed din_ice attention
      pooling = FLAGS.his_pooling if key in din_keys else FLAGS.his_pooling2
      self.his_poolings[key] = mt.layers.Pooling(pooling, name=f'his_{key}')
    for key in FLAGS.his_feats + FLAGS.his_feats2: #author,singer,song,tag...  use att attention as like tag might be Nan
      pooling = FLAGS.his_pooling if key in din_keys else FLAGS.his_pooling2
      self.his_poolings[key] = mt.layers.Pooling(pooling, name=f'his_{key}')
    for action in FLAGS.his_actions:
      self.his_poolings[action] = mt.layers.Pooling(FLAGS.his_pooling, name=f'his_{action}')
    for action in FLAGS.his_actions:
      for key in keys:
        pooling = FLAGS.his_pooling if key in din_keys else FLAGS.his_pooling2
        self.his_poolings[f'{key}_{action}'] = mt.layers.Pooling(pooling, name=f'his_{key}_{action}')
    for action in FLAGS.his_user_actions:
      self.his_poolings[action] = mt.layers.Pooling(FLAGS.his_pooling, name=f'his_u{action}')
    for action in FLAGS.his_user_actions:
      for key in user_keys:
        pooling = FLAGS.his_pooling if key in din_keys else FLAGS.his_pooling2
        self.his_poolings[f'{key}_{action}'] = mt.layers.Pooling(pooling, name=f'his_{key}_{action}')    
    if FLAGS.return_sequences:
      self.his_poolings[f'merge'] = mt.layers.Pooling(FLAGS.his_pooling, name=f'his_merge')
      self.merge_dense = keras.layers.Dense(FLAGS.emb_dim, name='merge_dense')
      self.merge_denses = {}
      for action in FLAGS.his_actions:
        self.his_poolings[f'merge_{action}'] = mt.layers.Pooling(FLAGS.his_pooling, name=f'his_merge_{action}')
        self.merge_denses[action] = keras.layers.Dense(FLAGS.emb_dim, name=f'merge_dense_{action}')
          
    self.his_encoder = HisEncoder(FLAGS.his_encoder, name='his_encoder')
    self.his_encoders = {}
    for key in FLAGS.his_actions:
      self.his_encoders[key] = HisEncoder(FLAGS.his_encoder, name=f'his_{key}_encoder')
    for key in keys:
      self.his_encoders[key] = HisEncoder(FLAGS.his_encoder, name=f'his_{key}_encoder')
    for action in FLAGS.his_actions:
      for key in keys:
        self.his_encoders[f'{key}_{action}'] = HisEncoder(FLAGS.his_encoder, name=f'his_{key}_{action}_encoder')
    for action in FLAGS.his_user_actions:
      for key in user_keys:
        self.his_encoders[f'{key}_{action}'] = HisEncoder(FLAGS.his_encoder, name=f'his_{key}_{action}_encoder')
    if FLAGS.return_sequences:
      self.doc_encoder = Encoder(pooling=FLAGS.doc_encoder_pooling, emb_dim=FLAGS.emb_dim, name='doc_encoder')
      self.doc_encoders = {}
      self.doc_his_encoder = SeqsEncoder(self.doc_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='doc_his_encoder')
      self.doc_his_encoders = {}
      pooling = FLAGS.seqs_pooling if not FLAGS.share_seqs_pooling else mt.layers.Pooling(FLAGS.seqs_pooling, name=f'doc_seqs_pooling')
      for action in FLAGS.his_actions:
        if FLAGS.share_doc_encoder:
          self.doc_his_encoders[action] =  SeqsEncoder(self.doc_encoder, FLAGS.seqs_encoder, pooling, name=f'doc_his_{action}_encoder')
        else:
          self.doc_encoders[action] = Encoder(pooling=FLAGS.doc_encoder_pooling, emb_dim=FLAGS.emb_dim, name=f'doc_{action}_encoder')
          self.doc_his_encoders[action] =  SeqsEncoder(self.doc_encoders[action], FLAGS.seqs_encoder, pooling, name=f'doc_his_{action}_encoder')

    # not used
    self.doc_pooling = mt.layers.Pooling(FLAGS.his_pooling2, name='Doc')
    self.user_pooling = mt.layers.Pooling(FLAGS.his_pooling2, name='User')
    self.context_pooling = mt.layers.Pooling(FLAGS.his_pooling2, name='Context')

    if FLAGS.share_tag_encoder:
      self.tag_his_encoder = SeqsEncoder(self.tag_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='tag_his_encoder')
    else:
      self.manual_tags_his_encoder = SeqsEncoder(self.manual_tags_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='manual_tags_his_encoder')
      self.machine_tags_his_encoder = SeqsEncoder(self.machine_tags_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='machine_tags_his_encoder')    

    if FLAGS.share_key_encoder:
      self.key_his_encoder = SeqsEncoder(self.key_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='key_his_encoder')
    else:
      self.manual_keys_his_encoder = SeqsEncoder(self.manual_keys_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='manual_keys_his_encoder')  
      self.machine_keys_his_encoder = SeqsEncoder(self.machine_keys_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='machine_keys_his_encoder')  

    if FLAGS.share_text_encoder:
      self.word_his_encoder = SeqsEncoder(self.word_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='word_his_encoder')
    else:
      self.desc_his_encoder = SeqsEncoder(self.desc_encoder, FLAGS.seqs_encoder, FLAGS.seqs_pooling, name='desc_his_encoder')

    self.embs_dict = {}

    Dense = mt.layers.MultiDropout if FLAGS.use_mdrop else keras.layers.Dense

    mlp_kwargs = {
      'drop_rate': FLAGS.dropout,
      'activation': FLAGS.mlp_activation,
      'batch_norm': FLAGS.mlp_batchnorm
    }
    if not FLAGS.task_mlp:
      dims = FLAGS.mlp_dims
      self.mlp = mt.layers.MLP(dims, name='mlp')
      dims2 = [dims[0], dims[1], FLAGS.emb_dim]
      self.dense_mlp = mt.layers.MLP(dims2, name='dense_mlp', **mlp_kwargs)
      self.dense = Dense(len(FLAGS.loss_list), name='dense')
    else:
      dims = FLAGS.mlp_dims[:2]
      self.mlp = mt.layers.MLP(dims, name='mlp', **mlp_kwargs)
      dims2 = [dims[0], dims[1], FLAGS.emb_dim]
      self.dense_mlp = mt.layers.MLP(dims2, name='dense_mlp', **mlp_kwargs)

      self.mlps = [] 
      self.denses = []
      for action in FLAGS.loss_list:
        self.mlps.append(mt.layers.MLP(FLAGS.task_mlp_dims, name=f'mlp_{action}', **mlp_kwargs))
        self.denses.append(Dense(1, name=f'dense_{action}'))

    self.pos_dense = Dense(1, name='pos_dense')
    self.neg_dense = Dense(1, name='neg_dense')

    if FLAGS.mmoe:
      self.mmoe = mt.layers.MMoE(
                                  len(FLAGS.loss_list), int(len(FLAGS.loss_list) * 1.5), 
                                  [512, FLAGS.emb_dim], 
                                  activation=FLAGS.mlp_activation,
                                )
      # # self.mmoe = mt.layers.MMoE(len(FLAGS.loss_list), len(FLAGS.loss_list), [512, FLAGS.emb_dim])
      # self.mmoe = mt.layers.MMoE(len(FLAGS.loss_list), 4, [512, FLAGS.emb_dim])

    self.emb_dropout = keras.layers.Dropout(FLAGS.emb_dropout)
    self.pooling_dropout = keras.layers.Dropout(FLAGS.pooling_dropout)
    self.batch_norm = keras.layers.BatchNormalization()
    self.layer_norm = keras.layers.LayerNormalization()

    if FLAGS.embs_encoder:
      self.embs_encoder = get_encoder(FLAGS.embs_encoder)

    if FLAGS.uncertain_loss:
      self.uncertain = mt.layers.UncertaintyLoss(len(FLAGS.loss_list))

    if FLAGS.sample_method:
      self.dots = keras.layers.Dot(axes=(2, 2))

    if FLAGS.cross_layers:
      self.cross = mt.layers.Cross(FLAGS.cross_layers)

    self.user_feats = ['user']
    self.first = True

  def deal_dense(self, input):
    feats = []
    if self.first:
      logging.ice('dese_feats', FLAGS.dense_feats)
    for dense_feat in FLAGS.dense_feats:
      max_val = 1.
      if 'stay_rate' in dense_feat:
        max_val = 2.
      # print(dense_feat, input[dense_feat])
      # feats.append(mt.scalar_feature(tf.squeeze(input[dense_feat], 1), max_val, scale=True))
      feats.append(mt.scalar_feature(tf.squeeze(input[dense_feat], 1)))
    
    if self.first:
      logging.ice('count_feats', FLAGS.count_feats)
    # 考虑更好的归一化处理 bin normalize https://tech.meituan.com/2018/03/29/recommend-dnn.html
    for count_feat in FLAGS.count_feats:
      # feats.append(mt.count_feature(input[count_feat]))
      max_val = 10000
      feats.append(mt.scalar_feature(tf.squeeze(input[count_feat], 1), max_val, scale=True))

    feats = tf.concat(feats, -1)
    # print(feats)
    dense_emb = self.dense_mlp(feats)
    return dense_emb

  def get_encoder(self, feat):
    if FLAGS.share_tag_encoder and 'tag' in feat:
      return self.tag_encoder
    if FLAGS.share_key_encoder and 'key' in feat:
      return self.key_encoder
    if FLAGS.share_text_encoder and ('desc' in feat or 'ocr' in feat or 'asr' in feat):
      if 'char' in feat:
        return self.char_encoder
      else:
        return self.word_encoder
    return getattr(self, f'{feat}_encoder')

  def get_his_encoder(self, feat):
    if FLAGS.share_tag_encoder and 'tag' in feat:
      return self.tag_his_encoder
    if FLAGS.share_key_encoder and 'key' in feat:
      return self.key_his_encoder
    if FLAGS.share_text_encoder and 'desc' in feat or 'ocr' in feat or 'asr' in feat:
      if 'char' in feat:
        return self.char_his_encoder
      else:
        return self.word_his_encoder
    return getattr(self, f'{feat}_his_encoder')

  def get_dense(self, key):
    if key in ['desc', 'ocr', 'asr']:
      return self.word_dense
    if 'char' in key:
      return self.char_dense
    else:
      return getattr(self, f'{key}_dense')

  def get_emb(self, key):
    return getattr(self, f'{key}_emb')

  def add_his_ids(self, input, his_id_feats, actions, user_actions,  
                  self_keys=[], return_keys=[],
                  return_sequences=False, return_sequences_only=False, training=None):
    add, adds = self.add, self.adds
    his_embs_dict = {}
    hlen = None
    
    for action in actions:
      his_embs_dict[action] = {}
    for action in user_actions:
       his_embs_dict[f'u_{action}'] = {}

    for key in his_id_feats:
      use_span = FLAGS.use_span and (key != 'user')
      emb_ = self.get_emb(key)

      try:
        dense_ = self.get_dense(key)
      except Exception:
        dense_ = None

      actions_ = actions if key != 'user' else user_actions
      for i, action in enumerate(actions_):
        indexes = input[action] if key != 'user' else input[f'u_{action}']
        if use_span:
          spans = input[f'{action}_spans']
        max_his = gezi.get('conf')['his_len'][action]
        if FLAGS.max_his:
          max_his = min(max_his, FLAGS.max_his)
        indexes = indexes[:, :max_his]
        if use_span:
          spans = spans[:, :max_his]
        hlen = mt.length(indexes)
        hlen = tf.math.maximum(hlen, FLAGS.his_min)  

        # ic(action, max_his, indexes.shape) 
        
        # if FLAGS.mask_his and (training or FLAGS.eval_mask_key):
        #   indexes = mask_key(indexes, key)

        his_embs = emb_(indexes)
        #his_embs = self.his_encoders[feat](his_embs, hlen)
        if his_embs.shape[-1] != FLAGS.emb_dim:
          his_embs = dense_(his_embs)

        if FLAGS.use_position:
          bs = mt.get_shape(his_embs, 0)
          postions = tf.tile(tf.expand_dims(tf.range(indexes.shape[-1]), 0),[bs, 1])
          position_embs = self.position_emb(postions)
          his_embs += position_embs

        if use_span:
          spans = tf.minimum(spans, FLAGS.max_his_days)
          span_embs = self.span_emb(spans)
          his_embs += span_embs
        
        if return_sequences and key != 'user' and ((not return_keys) or (key in return_keys)):
          his_embs_dict[action][key] = his_embs
          if return_sequences_only or (self_keys and (key not in self_keys)):
            continue

        user_feat = f'{key}_{action}'
        query = self.embs_dict[key] if key in self.embs_dict else None
        query_ = query
        if not FLAGS.encode_query:
          query_ = None
        his_encoder = self.his_encoders[user_feat] 
        if FLAGS.share_his_encoder:
          his_encoder = self.his_encoder
          his_embs = tf.concat([self.action_emb(tf.zeros_like(input['doc']) + i), his_embs], 1)
          hlen += 1
        if query_ is None:
          his_embs = his_encoder(his_embs, hlen)
        else:
          his_embs, query_ = his_encoder(his_embs, hlen, query_)
        if query_ is not None:
          query = query_
        assert query != None
        ## TODO FIXME save model get_model()时候会出错
        ## <melt.layers.layers.Pooling object at 0x7fcfafc80470> KerasTensor(type_spec=NoneTensorSpec(), description="created by layer 'his_encoder_15'")
        ## 也就是说 这个时候query是None 暂时log.warning改成log.debug  可能由于上面各种分支太复杂 tf判断构图不了了bug 如果直接 query = self.embs_dict[key] 是没有问题的
        ## 不过即使这样还是save graph 失败
        #  2021-07-28 07:07:22 0:00:28 Graph disconnected: cannot obtain value for tensor KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name='video_display'), 
        # name='video_display', description="created by layer 'video_display'") at layer "tf.compat.v1.squeeze_21". The following previous layers were accessed without issue: []
        # 2021-07-28 07:07:22 0:00:28 model.get_model(to functinal) fail
        his_pooling = self.his_poolings[user_feat] if not FLAGS.share_his_pooling else self.his_poolings[key]
        his_emb = his_pooling(his_embs, hlen, query)
        add(tf.expand_dims(his_emb, 1), f'his_ids_{user_feat}')   
        self.embs_dict[user_feat] = his_emb
        self.user_feats += [user_feat]

        if FLAGS.add_query_embs:
          if query is not None:
            add(tf.expand_dims(query, 1), f'his_ids_{user_feat}_query')

        if FLAGS.latest_actions and key != 'user':
          for i in range(FLAGS.latest_actions):
            if i < his_embs.shape[1]:
              user_feat = f'latest_{action}_{i}'
              add(tf.expand_dims(his_embs[:,i], 1), user_feat)
              self.embs_dict[user_feat] = his_emb
              self.user_feats += [user_feat] 
    
    return his_embs_dict, hlen

  def add_his_feats(self, input, his_feats, his_feats2, actions, 
                    self_keys=[], return_keys=[],
                    return_sequences=False, return_sequences_only=False, training=None):
    add, adds = self.add, self.adds
    his_embs_dict = {}
    hlen = None
    max_his = FLAGS.max_his2 or FLAGS.max_his
    if return_sequences:
      actions = FLAGS.his_actions
      max_his = FLAGS.max_his
    
    if his_feats or his_feats2:
      for i, action in enumerate(actions):
        his_embs_dict[action] = {}
        indexes = input[action]
        if max_his:
          indexes = indexes[:, :max_his]
        hlen = mt.length(indexes)
        hlen = tf.math.maximum(hlen, FLAGS.his_min)    

        for feat in his_feats:
          his_ = tf.squeeze(util.lookup(feat, indexes), -1)  

          # if FLAGS.mask_his and feat in FLAGS.mask_keys and (training or FLAGS.eval_mask_key):  
          #   his_ = mask_key(his_, feat)

          his_embs = self.embs[feat](his_)

          if return_sequences and ((not return_keys) or (feat in return_keys)):
            his_embs_dict[action][feat] = his_embs
            if return_sequences_only or (self_keys and (feat not in self_keys)):
              continue

          # TODO 包括his_feats, his_feats2 是否都考虑使用HisEncoder encoder考虑历史+当前整体建模?

          his_emb = self.his_poolings[feat](his_embs, hlen, self.embs_dict[feat])
          user_feat = f'{feat}_{action}'
          add(tf.expand_dims(his_emb, 1), f'his_feat_{user_feat}')
          self.embs_dict[user_feat] = his_emb
          self.user_feats += [user_feat]

        for feat in his_feats2:
          # print(action, feat, his[feat])
          his_embs = util.lookup(feat, indexes)
          hlen_ = hlen
          feat_ = feat
          
          max_his = FLAGS.max_his3
          if not 'tag' in feat and not 'key' in feat:
            max_his = FLAGS.max_texts
          if return_sequences:
            max_his = FLAGS.max_his
          if max_his:
            his_embs = his_embs[:, :max_his]
            hlen_ = tf.math.minimum(hlen, max_his)

          encoder = self.get_his_encoder(feat)
          if feat == 'machine_tags' and FLAGS.machine_weights:
            his_embs = tf.concat([his_embs, input['machine_tag_probs']], -1)

          his_emb = encoder(his_embs, hlen_, self.embs_dict[feat], return_sequences=return_sequences)
          
          if (his_emb.shape[-1] != FLAGS.emb_dim) or (feat in FLAGS.dense_transform_keys):
            try:
              dense = self.get_dense(feat)
              his_emb = dense(his_emb)
            except Exception:
              logging.error(f'no dense for {feat}', his_emb.shape[-1], FLAGS.emb_dim)
              pass

          if return_sequences and ((not return_keys) or (feat in return_keys)):
            his_embs_dict[action][feat] = his_emb
            if return_sequences_only or (self_keys and (feat not in self_keys)):
              continue

          user_feat = f'{action}_{feat}'
          add(tf.expand_dims(his_emb, 1), f'his_feat2_{user_feat}')
          self.embs_dict[user_feat] = his_emb
          self.user_feats += [user_feat]

    return his_embs_dict, hlen

  def call(self, input, training=None):
    # 注意需要cpy 否则下面有其他操作可能会 The list of inputs passed to the model is redundant. All inputs should only appear once.
    self.input_ = input.copy()

    if FLAGS.tower_train:
      user_embs, doc_embs, context_embs = self.forward(input, training=training)
      perm = tf.random.shuffle(tf.range(tf.shape(user_embs)[0]))
      input = input.copy()
      for key in input:
        input[key] = tf.gather(input[key], perm, axis=0)
      _, doc_embs2, _ = self.forward(input, training=training)
      # embs_pos = self.pooling(tf.concat([user_embs, context_embs, doc_embs], 1))
      # embs_neg = self.pooling(tf.concat([user_embs, context_embs, doc_embs2], 1))
      embs_pos = self.pooling(tf.concat([user_embs, doc_embs], 1))
      embs_neg = self.pooling(tf.concat([user_embs, doc_embs2], 1))
      logit_pos = self.pos_dense(embs_pos)
      logit_neg = self.neg_dense(embs_neg)
      logit = tf.stack([logit_pos, logit_neg], 1)
      # print(logit)
      return logit
    elif FLAGS.rdrop_loss:
      self.pred1 = self.forward(input, training=training)
      if training:
        self.pred2 = self.forward(input, training=training)
      else:
        self.pred2 = self.pred1
      return self.pred1
    else:
      return self.forward(input, training=training)

  def forward(self, input, training=None):
    add, adds = self.add, self.adds
    self.clear()

    # print(input['poss'])
    # print(input['num_poss'])

    for name in FLAGS.doc_keys:
      input[name] = input['doc']

    #TODO 很奇怪fp16 stay 有很多inf
    #  [4.5080e+03],  4508 正常
    #  [6.7729e+04],  会成为inf 。。。
    # 尝试functional model会报错 doc 名字用了两次。。 TODO
    # bs = mt.get_shape(input['feed'], 0)
    # return tf.ones([bs, 4])
    
    for key in get_info_keys():
      if key not in input:
        # singer (bs, 1, 1) -> (bs, 1)  desc (bs, 1, 64) -> (bs, 64)
        input[key] = tf.squeeze(util.lookup(key, input['doc']), 1)

    if FLAGS.mask_his_rate > 0 and training:
      for action in FLAGS.his_actions:
        input[action] = aug(input[action], FLAGS.mask_his_rate)

    embs = self.feats
    if FLAGS.add_action_embs:
      for i, action in enumerate(ACTIONS):
        add(self.action_emb(tf.zeros_like(input['doc']) + i), action)

    # single doc feature
    for feat in FLAGS.feats:
      if feat == 'doc' and not FLAGS.use_doc_emb:
        continue
      if feat == 'user' and not FLAGS.use_user_emb:
        continue
      indexes = input[feat]

      if feat in FLAGS.mask_keys and (training or FLAGS.eval_mask_key):
        indexes = mask_key(indexes, feat)

      if FLAGS.map_unk and feat in unk_keys:
        indexes1 = indexes
        indexes = map_unk(indexes)

      emb = self.embs[feat](indexes)
      if (not training) and FLAGS.mean_unk:
        emb = mean_unk(emb)

      if feat == 'feed' and FLAGS.concat_feed:
        self.embs_dict['feed_ori'] = tf.squeeze(emb, 1)
      
      if emb.shape[-1] != FLAGS.emb_dim:
        dense = self.get_dense(feat)
        emb = dense(emb)
      add(emb, f'feat_{feat}')
      # tf 2.3 v100 tione环境很奇怪 fp16 embeding lookup出来还是float32 feed由于过了Dense 变成float16 tf2.4.1似乎没问题
#       print(feat, emb.shape, emb.dtype)
      self.embs_dict[feat] = tf.squeeze(emb, 1)

    # multi doc feature
    for feat in FLAGS.feats2:
      encoder = self.get_encoder(feat)
      ifeat = input[feat] 

      if FLAGS.map_unk and feat in unk_keys:
        ifeat = map_unk(ifeat)

      if feat == 'machine_tags' and FLAGS.machine_weights:
        ifeat = tf.concat([ifeat, input['machine_tag_probs']], -1)

      emb = encoder(ifeat)
      if emb.shape[-1] != FLAGS.emb_dim:
        dense = self.get_dense(feat)
        emb = dense(emb)
      add(tf.expand_dims(emb, 1), f'feat2_{feat}')
      self.embs_dict[feat] = emb
    
    return_sequences, return_sequences_only = FLAGS.return_seqs, FLAGS.return_seqs_only
    his_embs_dict1, hlen = self.add_his_ids(input, FLAGS.his_id_feats, 
                                            FLAGS.his_actions, FLAGS.his_user_actions,
                                            FLAGS.id_self_keys, FLAGS.id_return_keys,
                                            return_sequences, return_sequences_only, training)
    his_embs_dict2, _ = self.add_his_feats(input, FLAGS.his_feats, FLAGS.his_feats2, 
                                           FLAGS.his_actions2 or FLAGS.his_actions,  
                                           FLAGS.id_self_keys2, FLAGS.id_return_keys2,
                                           return_sequences, return_sequences_only, training)

    if return_sequences:
      for i, action in enumerate(FLAGS.his_actions):
        cur_embs = []
        his_embs = []
        keys = []
        if his_embs_dict1:
          for key in his_embs_dict1[action]:
            cur_embs.append(self.embs_dict[key])
            his_embs.append(his_embs_dict1[action][key])
            keys.append(key)
        if his_embs_dict2:
          for key in his_embs_dict2[action]:
            cur_embs.append(self.embs_dict[key])
            his_embs.append(his_embs_dict2[action][key])
            keys.append(key)

        num_embs = len(his_embs)
        if self.first and i == 0:
          logging.ice('num_merge_embs:', num_embs, keys)
        if num_embs > 1:
          num_feats = len(his_embs)
          if not FLAGS.doc_his_encoder:
            # TODO 尝试替换为encoder 和 his_encoder这样多个维度输入的融合可能效果更好
            his_embs = tf.concat(his_embs, -1)
            merge_dense = self.merge_dense if FLAGS.share_doc_encoder else self.merge_denses[action]
            his_embs = merge_dense(his_embs)
            if not FLAGS.share_doc_encoder or i == 0:
              cur_emb = tf.concat(cur_embs, -1)
              cur_emb = merge_dense(cur_emb)
            if FLAGS.share_doc_encoder:
              if i == 0:
                add(tf.expand_dims(cur_emb, 1), 'merge_cur_doc')
            else:
              add(tf.expand_dims(cur_emb, 1), f'merge_{action}_doc')
            his_embs = self.his_encoders[action](his_embs, hlen)
            his_pooling = self.his_poolings[action] if not FLAGS.share_his_pooling else self.his_poolings['merge']
            hemb = his_pooling(his_embs, hlen, cur_emb)
            user_feat = f'his_ids_merge_{action}'
            add(tf.expand_dims(hemb, 1), user_feat)   
            self.embs_dict[user_feat] = hemb
            self.user_feats += [user_feat]
          else:
            # here with doc_encoder_pooling=dense should be the same as above merge version
            # not tested online offline 713 verse 714 still a bit lower, not sure why
            his_embs = tf.stack(his_embs, 2)
            # ic(his_embs.shape)
            doc_encoder = self.doc_encoder if FLAGS.share_doc_encoder else self.doc_encoders[action]
            if not FLAGS.share_doc_encoder or i == 0:
              # (bs, 2, 128)
              cur_emb = tf.stack(cur_embs, 1)
              # ic(cur_emb.shape)
              cur_emb = doc_encoder(cur_emb)
              # ic(cur_emb.shape)
            if FLAGS.share_doc_encoder:
              if i == 0:
                add(tf.expand_dims(cur_emb, 1), 'encode_cur_doc')
            else:
              add(tf.expand_dims(cur_emb, 1), f'encode_{action}_doc')
            # ic(his_embs.shape, hlen, cur_emb.shape)
            if FLAGS.share_doc_his_encoder:
              assert FLAGS.share_doc_encoder
              # his_embs = tf.concat([self.action_emb(tf.zeros_like(input['doc']) + i), his_embs], 1)
              hemb = self.doc_his_encoder(his_embs, hlen, cur_emb)
            else:
              hemb = self.doc_his_encoders[action](his_embs, hlen, cur_emb)
            user_feat = f'his_ids_encode_{action}'
            add(tf.expand_dims(hemb, 1), user_feat)   
            self.embs_dict[user_feat] = hemb
            self.user_feats += [user_feat]

    if FLAGS.use_tower:
      doc_embs = []
      user_embs = []
      context_embs = []
      for key, emb_ in self.embs_dict.items():
        if key in DOC_FEATS:
          doc_embs += [tf.expand_dims(emb_, 1)]
        elif key in self.user_feats:
          user_embs += [tf.expand_dims(emb_, 1)]
        else:
          context_embs += [tf.expand_dims(emb_, 1)]
      doc_embs = tf.concat(doc_embs, 1)
      user_embs = tf.concat(user_embs, 1)
      context_embs =tf.concat(context_embs, 1)

      if FLAGS.tower_train:
        return doc_embs, user_embs, context_embs

      add(tf.expand_dims(self.user_pooling(user_embs), 1), 'user_feat')
      add(tf.expand_dims(self.doc_pooling(doc_embs), 1), 'doc_feat')
      add(tf.expand_dims(self.context_pooling(context_embs), 1), 'context_feat')

    if FLAGS.use_dense:
      dense_emb = self.deal_dense(input)
      add(tf.expand_dims(dense_emb, 1), 'dense_feat')

    # print([(i, x.shape) for i, x in enumerate(embs)])
    # exit(0)
    
    if self.first:
      self.print_feats(logging.ice)

    # TODO HACK For tf2.3 fp16 https://github.com/tensorflow/tensorflow/issues/41614 embdding lookup出来是float32 可能2.4之后修改了
    # 另外这里fp16 反而更慢.. 对应tione v100 tf2.3环境 
    if (not FLAGS.fp16) or tf.__version__ >= '2.4':
      embs = tf.concat(embs, 1)
    else:
      embs = tf.concat([tf.cast(emb, tf.float16) for emb in embs], 1)

    if FLAGS.emb_stop_rate < 1.:
      embs = FLAGS.emb_stop_rate * embs + (1 - FLAGS.emb_stop_rate) * tf.stop_gradient(embs)

    # ori_embs = embs
    if self.first:
      logging.ice('embs.shape', embs.shape)

    if FLAGS.add_l2_dense:
      l2s = tf.math.log(tf.reduce_sum(embs * embs, axis=-1) + 1)
      # pns = tf.cast(tf.reduce_sum(embs, axis=-1) > 0, embs.dtype)

    self.ori_embs = embs
    assert FLAGS.pooling or FLAGS.pooling2
    # 第一次pooling之前 不做任何预处理
    if FLAGS.pooling:
      if FLAGS.embs_encoder and FLAGS.embs_encoder_position == 1:
        embs = self.embs_encoder(embs)
      x = self.pooling(embs)
      self.x_pooling = x
      if self.first:
        logging.ice('after_pooling.shape', x.shape)
    else:
      x = None

    if FLAGS.cross_layers:
      x = self.cross(x)

    # 第二个pooling 做l2 norm 计算dot实际变成了cosine相似度 效果更好
    if FLAGS.pooling2:
      if FLAGS.pooling2_ori_embs:
        embs = self.ori_embs
      embs = self.emb_dropout(embs)
      if FLAGS.batch_norm:
        embs = self.batch_norm(embs, training=training)
      if FLAGS.layer_norm:
        embs = self.layer_norm(embs, training=training)
      if FLAGS.embs_encoder and FLAGS.embs_encoder_position == 2:
        embs = self.embs_encoder(embs)
      if FLAGS.l2_norm:
        embs = tf.math.l2_normalize(embs, axis=-1)
      self.normed_embs = embs
      x2 = self.pooling2(embs)
      self.x_pooling2 = x2
      # cosine特征放最左侧
      x = x2 if x is None else tf.concat([x2, x], -1)

    if self.first:
      logging.ice('after_pooling2.shape', x.shape)

    if FLAGS.concat_feed:
      x = tf.concat([x, self.embs_dict['feed_ori']], -1)

    x = self.pooling_dropout(x)

    if FLAGS.add_l2_dense:
      x = tf.concat([x, l2s], -1)

    # if FLAGS.use_dense:
    #   x = tf.concat([x, dense_emb], -1)

    if self.first:
      logging.ice('final.shape', x.shape)

    if FLAGS.mmoe:
      xs = self.mmoe(x)
      logits = []
      for i in range(len(FLAGS.loss_list)):
        x_ = xs[i]
        if FLAGS.add_action_embs:
          action = ACTIONS[i]
          # ic(action, self.feats_[action])
          x_ = tf.concat([x_, embs[:,i]], -1)
        if not FLAGS.mmoe_mlp:
          logits.append(self.denses[i](x_))
        else:
          logits.append(self.denses[i](self.mlps[i](x_)))
      logit = tf.concat(logits, -1)
    else:
      if not FLAGS.task_mlp:
        x = self.mlp(x)
        logit = self.dense(x)
      else:
        logits = []
        for i in range(len(FLAGS.loss_list)):
          x_ = x
          if FLAGS.add_action_embs:
            action = ACTIONS[i]
            x_ = tf.concat([x_, embs[:,i]], -1)
          logits.append(self.denses[i](self.mlps[i](x_)))
        logit = tf.concat(logits, -1)
    if self.first:
      logging.ice('final logit.shape', logit.shape)  
    self.prob = tf.nn.sigmoid(logit)

    if FLAGS.sample_method:
      # ic(input['context'])
      feed_emb = self.feats_['feat_feed']
      his_emb = self.feats_['his_ids_feed_poss']
      feed_embs = tf.concat([his_emb, self.feed_emb(input['context'][:,1:])], 1)
      self.dot = tf.squeeze(self.dots([feed_embs, feed_emb]), -1)
    
    self.first = False 
    self.logit = logit
    if FLAGS.action_loss:
      self.prob = self.prob[:,-1:] * self.prob[:,:-1]
      logit = mt.prob2logit(self.prob)
    return logit

  def get_loss(self):
    from wechat import loss
    if FLAGS.tower_train:
      loss_fn_ = loss.tower_loss_fn
    elif FLAGS.rdrop_loss:
      loss_fn_ = loss.rdrop_loss
    else:
      loss_fn_ = loss.get_loss_fn()
    return self.loss_wrapper(loss_fn_)

def get_model(model_name):
  import wechat
  from .dataset import Dataset
  if model_name == 'None':
    return mt.Model()

  model = getattr(wechat.model, model_name)() 
  if FLAGS.functional_model:
    model = mt.to_functional_model(model, Dataset)

  return model
