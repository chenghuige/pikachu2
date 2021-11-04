#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   bert.py
#        \author   chenghuige  
#          \date   2020-09-02 16:08:18.343557
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input

# from third.bert import modeling
from gezi import logging
import melt

if tf.__version__ < '2':
  # bert fot tf2 兼容1，2 但是参数载入貌似不能直接在__init__做? 需要input 写法有点怪 
  import bert # https://github.com/kpe/bert-for-tf2

  class Bert(keras.Model):
    def __init__(self, model_dir, output_dim=None, max_input_length=None, lr_rate=0.002,
                 trainable=True, return_sequences=False, **kwargs):
      super(Bert, self).__init__(**kwargs)

      self.model_dir = model_dir
      self.max_input_length = max_input_length
      self.lr_rate = lr_rate
      self.return_sequences = return_sequences
      if model_dir:
        # try:
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.model = bert.BertModelLayer.from_params(bert_params, name="bert")
        self.model.trainable = trainable

        self.output_dim = output_dim
        bert_output_dim = bert_params['hidden_size']
        if bert_output_dim == self.output_dim or not output_dim:
          self.proj = lambda x: x  
        else:
          self.proj = keras.layers.Dense(output_dim)
        if not self.output_dim:
          self.output_dim = bert_output_dim

        self.inited = False
        # except Exception:
        #   pass

    def compute_output_shape(self, input_shape):
      return (None, self.output_dim)

    def build(self, input_shape):
      # TODO use tf.train.latest_checkpoint(ckpt_dir) ?
      pass

    def call(self, x, seq_len=None, return_sequences=False):
      if not self.inited:
        self.model(x)
        bert_ckpt_file = os.path.join(self.model_dir, "bert_model.ckpt")
        bert.load_stock_weights(self.model, bert_ckpt_file)
        self.inited = True

      x = self.model(x)
      if not (self.return_sequences and return_sequences):
        x = x[:, 0]

      x = x * self.lr_rate + tf.stop_gradient(x) * (1 - self.lr_rate)
      return self.proj(x)

else:
  # transformers 只支持tf2 
  # 通过trasformers的命令行可以把 tf1版本google开源的训练代码生成的模型转换成pytorch再在tf2载入
  # transformers-cli convert --tf_checkpoint ./bert_model.ckpt  --pytorch_dump_output ./pytorch_model.bin --config ./bert_config.json --model_type bert
  from transformers import BertConfig, TFBertModel, AutoConfig, TFAutoModel
  class Bert(keras.Model):
    def __init__(self, model_dir, output_dim=None, max_input_length=None, lr_rate=1.,
                 trainable=True, dropout=0., return_sequences=False, from_pt=False, **kwargs):
      super(Bert, self).__init__(**kwargs)

      self.model_dir = model_dir
      self.max_input_length = max_input_length
      self.lr_rate = lr_rate
      self.return_sequences = return_sequences
      if model_dir:
        if not os.path.exists(model_dir):
          self.model = TFAutoModel.from_pretrained(model_dir, from_pt=from_pt)
          config = AutoConfig.from_pretrained(model_dir)
        else:
          config = BertConfig.from_json_file(f'{model_dir}/bert_config.json')
          self.model = TFBertModel.from_pretrained(model_dir, from_pt=from_pt, config=config)
        self.model.trainable = trainable
        if max_input_length:
          inputs = Input(shape=(max_input_length,), dtype=tf.int32, name="input_word_ids")
          out = self.model(inputs)
          self.model = tf.keras.Model(inputs=inputs, outputs=out)

        self.output_dim = output_dim
        bert_output_dim = config.hidden_size
        if bert_output_dim == self.output_dim or not output_dim:
          self.proj = lambda x: x  
        else:
          self.proj = melt.layers.Project(output_dim)
        if not self.output_dim:
          self.output_dim = bert_output_dim
      self.dropout = tf.keras.layers.Dropout(dropout) if dropout else lambda x: x

    def compute_output_shape(self, input_shape):
      return input_shape[:-1].concatenate(self.output_dim)

    def build(self, input_shape):
      # TODO use tf.train.latest_checkpoint(ckpt_dir) ?
      pass

    ## TODO not work... https://www.jiqizhixin.com/articles/2020-05-09-13
    # @melt.recompute_grad(True)
    def call(self, x, segment_ids=None, seq_len=None, return_sequences=False):
      if self.max_input_length:
        x = melt.pad(x, self.max_input_length)

      input_mask = tf.cast(x > 0, x.dtype) 
      x = self.model(x, input_mask, segment_ids)
      if not (self.return_sequences and return_sequences):
        x = x[0][:, 0]
        # try:
        #   x = x.pooler_output
        # except Exception:
        #   x = x.last_hidden_state[:,0]
      else:
        x = x[0]

      if self.lr_rate != None and self.lr_rate < 1.:
        x = x * self.lr_rate + tf.stop_gradient(x) * (1 - self.lr_rate)
      return self.proj(self.dropout(x))
      # return self.dropout(self.proj(x))
