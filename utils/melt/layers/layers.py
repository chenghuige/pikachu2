#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   layers.py
#        \author   chenghuige  
#          \date   2016-08-19 23:22:44.032101
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.lib.arraysetops import isin

import tensorflow as tf

import sys
from absl import flags
FLAGS = flags.FLAGS

import functools
import six
import re
from functools import partial
import traceback
import copy
from sklearn.preprocessing import normalize
from icecream import ic

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
import tensorflow_recommenders as tfrs

import numpy as np

import gezi
import melt

logging = gezi.logging

from melt import dropout, softmax_mask
from melt.rnn import OutputMethod, encode_outputs

# TODO -------------------------
# just use below really layers!
# TODO batch_dim to batch_axis

keras = tf.keras
layers = tf.keras.layers
Layer = layers.Layer

# https://www.jianshu.com/p/73b6f5d00f46
# https://zhuanlan.zhihu.com/p/78829402
class DiceActivation(Layer):
  def __init__(self, epsilon=1e-7, **kwargs):
    super(DiceActivation, self).__init__(**kwargs)
    self.norm = tf.keras.layers.BatchNormalization(
      axis=-1, epsilon=epsilon, center=False, scale=False)
   
  def build(self, input_shape):
    self.alpha = self.add_weight(name='alpha',
                                  shape=(input_shape[-1],),
                                  initializer='zeros',                         
                                  dtype=tf.float32,
                                  trainable=True)
    self.built = True
  
  def call(self, x):
    inputs_normed = self.norm(x)
    x_p = tf.sigmoid(inputs_normed)
    return self.alpha * (1.0 - x_p) * x + x_p * x
  
  def get_config(self):
    config = {
      'norm': self.norm,
      'alpha': self.alpha,
    }
    base_config = super(DiceActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class PreluActivation(Layer):
  def __init__(self, **kwargs):
    super(PreluActivation, self).__init__(**kwargs)
   
  def build(self, input_shape):
    self.alpha = self.add_weight(name='alpha',
                                  shape=(input_shape[-1],),
                                  initializer=tf.constant_initializer(0.1),                         
                                  dtype=tf.float32,
                                  trainable=True)
  
  def call(self, x):
    return tf.maximum(0.0, x) + self.alpha * tf.minimum(0.0, x)

def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3.0))))

def activation_layer(activation):
  if activation is None or activation == '':
    return None
  if (isinstance(activation, str)) or (sys.version_info.major == 2 and isinstance(activation, (str, unicode))):
    if activation == 'dice':
      return DiceActivation()
    if activation == 'prelu':
      return PreluActivation()
    if activation == 'grelu':
      return gelu
    act_layer = tf.keras.layers.Activation(activation)
  try:
    if issubclass(activation, Layer):
      act_layer = activation()
  except Exception:
    act_layer = activation
  return act_layer

class FeedForwardNetwork(Layer):
  def __init__(self, hidden_size, output_size, activation='relu', drop_rate=0.):
    super(FeedForwardNetwork, self).__init__()
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.linear1 = layers.Dense(hidden_size, activation=activation_layer(activation))
    self.linear2 = layers.Dense(output_size)
    self.dropout = layers.Dropout(drop_rate)

  def call(self, x):
    x_proj = self.dropout(self.linear1(x))
    x_proj = self.linear2(x_proj)
    return x_proj

class Project(Layer):
  """
  following drlm https://github.com/facebookresearch/dlrm/blob/master/tricks/md_embedding_bag.py
  """
  def __init__(self, units, **kwargs):
    super(Project, self).__init__(**kwargs)
    # 事实上kernel_initializer='glorot_uniform' 是 Dense的默认 tf2.3 等价 torch.nn.init.xavier_uniform_
    initializer = tf.keras.initializers.GlorotUniform()
    self.proj = layers.Dense(units, use_bias=False, kernel_initializer=initializer)

  def call(self, x):
    return self.proj(x)
            
class MaxPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1, reduce_func=tf.reduce_max):
    return melt.max_pooling(outputs, sequence_length, axis, reduce_func)

class SumPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1, reduce_func=tf.reduce_sum):
    return melt.sum_pooling(outputs, sequence_length, axis)

class MaxPooling2(Layer):
  def call(self, outputs, sequence_length, sequence_length2, axis=1, reduce_func=tf.reduce_max):
    return melt.max_pooling2(outputs, sequence_length, sequence_length2, axis, reduce_func)

class MeanPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return melt.mean_pooling(outputs, sequence_length, axis)

class SqrtnPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return melt.sqrtn_pooling(outputs, sequence_length, axis)

class FirstPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return outputs[:, 0, :]

class LastPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return melt.dynamic_last_relevant(outputs, sequence_length)

class HierEncode(Layer):
  def call(self, outputs, sequence_length=None, window_size=3, axis=1):
    return melt.hier_encode(outputs, sequence_length, window_size=3, axis=1)

class TopKPooling(Layer):
  def __init__(self,  
               top_k,
               **kwargs):
    super(TopKPooling, self).__init__(**kwargs)
    self.top_k = top_k
  
  def call(self, outputs, sequence_length=None, axis=1):
    if sequence_length is None:
      sequence_length = tf.reduce_sum(tf.ones_like(outputs, dtype=tf.int64))
    x = melt.top_k_pooling(outputs, self.top_k, sequence_length, axis).values  
    return tf.reshape(x, [-1, melt.get_shape(outputs, -1) * self.top_k])

class TopKMeanPooling(Layer):
  def __init__(self,  
               top_k,
               **kwargs):
    super(TopKMeanPooling, self).__init__(**kwargs)
    assert top_k > 1
    self.top_k = top_k
  
  def call(self, outputs, sequence_length=None, axis=1):
    x = melt.top_k_pooling(outputs, self.top_k, sequence_length, axis).values  
    x = tf.reduce_mean(input_tensor=x, axis=-1)
    return x

# not good..
class TopKWeightedMeanPooling(Layer):
  def __init__(self,  
               top_k,
               ratio=0.7,
               **kwargs):
    super(TopKWeightedMeanPooling, self).__init__(**kwargs)
    assert top_k > 1
    self.top_k = top_k
    self.w = [1.] * self.top_k
    for i in range(top_k - 1):
      self.w[i + 1] = self.w[i]
      self.w[i] *= ratio
      self.w[i + 1] *= (1 - ratio)
    self.w = tf.constant(self.w)
  
  def call(self, outputs, sequence_length=None, axis=1):
    x = melt.top_k_pooling(outputs, self.top_k, sequence_length, axis).values  
    x = tf.reduce_sum(input_tensor=x * self.w, axis=-1)
    return x

class TopKAttentionPooling(Layer):
  def __init__(self,  
               top_k,
               **kwargs):
    super(TopKAttentionPooling, self).__init__(**kwargs)
    assert top_k > 1
    self.top_k = top_k
    self.att = AttentionPooling()

  def call(self, outputs, sequence_length=None, axis=1):
    x = melt.top_k_pooling(outputs, self.top_k, sequence_length, axis).values  
    x = tf.transpose(a=x, perm=[0, 2, 1])
    x = self.att(x)
    return x

# TODO check which is better tf.nn.tanh or tf.nn.relu, by paper default should be tanh
# TODO check your topk,att cases before use relu.. seems tanh worse then relu, almost eqaul but relu a bit better and stable
# TODO 去掉到hiddn size的Dense 如果输入输出保持一致的话？
class AttentionPooling(Layer):
  def __init__(self,  
               hidden_size=None,
               activation=tf.nn.relu,
               transform_inputs=True,
               num_outputs=1,
               **kwargs):
    super(AttentionPooling, self).__init__(**kwargs)
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.activation = activation
    if hidden_size is not None:
      assert transform_inputs
      self.dense = layers.Dense(hidden_size, activation=activation_layer(activation))
    else:
      self.dense = None
    self.logits = layers.Dense(num_outputs)
    self.transform_inputs = transform_inputs

  def build(self, input_shape):
    if self.dense is None:
      if self.transform_inputs:
        self.dense = layers.Dense(input_shape[-1], activation=activation_layer(self.activation))
      else:
        self.dense = lambda x: x

  # TODO use build to setup self.dense
  def call(self, outputs, sequence_length=None, axis=1):
    x = self.dense(outputs)
    logits = self.logits(x)
    alphas = tf.nn.softmax(logits, axis=1) if sequence_length is None else  melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(outputs * alphas, axis=1)
    # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    # self.alphas = tf.squeeze(alphas, -1)    
    #self.alphas = alphas
    # tf.compat.v1.add_to_collection('self_attention', self.alphas) 
    return encoding

  def get_config(self):
    config = {
      'activation': self.activation,
      'desne': self.dense,
      'logits': self.logits
    }
    base_config = super(AttentionPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class LinearAttentionPooling(Layer):
  def __init__(self,  
               num_outputs=1,
               **kwargs):
    super(LinearAttentionPooling, self).__init__(**kwargs)
    self.logits = layers.Dense(num_outputs)

  def call(self, x, sequence_length=None, axis=1):
    logits = self.logits(x)
    alphas = tf.nn.softmax(logits, axis=1) if sequence_length is None else  melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(x * alphas, axis=1)
    # # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    # self.alphas = tf.squeeze(alphas, -1)    
    # #self.alphas = alphas
    # tf.compat.v1.add_to_collection('self_attention', self.alphas) 
    return encoding

class NonLinearAttentionPooling(Layer):
  def __init__(self,  
               hidden_size=None,
               **kwargs):
    super(NonLinearAttentionPooling, self).__init__(**kwargs)
    if hidden_size is not None:
      self.FFN = FeedForwardNetwork(hidden_size, 1)
    else:
      self.FFN = None

  def build(self, input_shape):
    if self.FFN is None:
      self.FFN = FeedForwardNetwork(input_shape[-1], 1)

  def call(self, x, sequence_length=None, axis=1):
    logits = self.FFN(x)
    alphas = tf.nn.softmax(logits) if sequence_length is None else  melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(x * alphas, axis=1)
    # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    self.alphas = tf.squeeze(alphas, -1)    
    #self.alphas = alphas
    tf.compat.v1.add_to_collection('self_attention', self.alphas) 
    return encoding

class ConcatPooling(Layer):
  def __init__(self,  
               **kwargs):
    super(ConcatPooling, self).__init__(**kwargs)
  
  def call(self, x, sequence_length=None, axis=1):
    shape = melt.get_shape(x)
    return tf.reshape(x, [-1, shape[axis] * shape[axis + 1]])

  def compute_output_shape(self, input_shape):
    out_shape = input_shape[:-1].concatenate(input_shape[-2] * input_shape[-1])
    return out_shape

  def get_config(self):
    config = {}
    base_config = super(ConcatPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class MLPPooling(Layer):
  def __init__(self,  
               dims=[],
               **kwargs):
    super(MLPPooling, self).__init__(**kwargs)
    self.mlp = None if not dims else MLP(dims, activate_last=False)
  
  def build(self, input_shape):
    if not self.mlp:
      times = input_shape[-2]
      odim = input_shape[-1]
#       odim1 = odim if times == 1 else int(times * odim / 2.)
      odim1 = odim * 2
      dims = [odim1, odim]
      self.mlp = MLP(dims, activate_last=False)

  def call(self, x, sequence_length=None, axis=1):
    shape = melt.get_shape(x)
    x = tf.reshape(x, [-1, shape[axis] * shape[axis + 1]])
    res = self.mlp(x)
    return res

class DensePooling(Layer):
  def __init__(self,  
               **kwargs):
    super(DensePooling, self).__init__(**kwargs)
    self.dense = None 
  
  def build(self, input_shape):
    if not self.dense:
      odim = input_shape[-1]
      self.dense = keras.layers.Dense(odim)

  def call(self, x, sequence_length=None, axis=1):
    shape = melt.get_shape(x)
    x = tf.reshape(x, [-1, shape[axis] * shape[axis + 1]])
    res = self.dense(x)
    return res

class SFUPooling(Layer):
  def __init__(self,  
               **kwargs):
    super(SFUPooling, self).__init__(**kwargs)  
    self.sfu = SemanticFusionCombine()

  def call(self, x, sequence_length=None, axis=1):
    xs = tf.split(x, 2, axis=1)
    xs = [tf.squeeze(x, 1) for x in xs]
    res = self.sfu(xs[0], xs[1])
    return res

class DotPooling(Layer):
  def __init__(self, interaction_itself=False, remove_duplicate=False, sqrtn_norm=False, l2_norm=False, **kwargs):
    super(DotPooling, self).__init__(**kwargs)
    # 这里注意和DotInteractionPooling相比 如果remove_uplicate = False,实际上interaction_itself选项不起作用 都做了自交叉
    # DotInteractionPooling逻辑是完全正确的, 并且做了变0处理 
    self.interaction_itself = interaction_itself
    self.remove_duplicate = remove_duplicate
    self.sqrtn_norm = sqrtn_norm
    self.l2_norm = l2_norm

  def call(self, x, sequence_length=None, axis=1):
    shape = melt.get_shape(x)
    # print(x, shape)
    x = tf.matmul(x, tf.transpose(x, [0, 2, 1]))
    if self.sqrtn_norm:
      dk = tf.cast(shape[-1], tf.float32)
      x = x / tf.math.sqrt(dk)
    if self.l2_norm:
      x = tf.nn.l2_normalize(x, axis=-1)
    gezi.set('dot_mat', x)
    if not self.remove_duplicate:
      return  tf.reshape(x, (-1, shape[axis] * shape[axis]))
    else:
      ## less output but slow, so prefere to use not remove_duplicate
      # x += 1e10  # bad result ..
      zero_mask = tf.cast(K.equal(x, 0), tf.float32)
      x += zero_mask * K.epsilon()
      x = tf.linalg.band_part(x, 0, -1)
      if not self.interaction_itself:
        x -= tf.linalg.band_part(x, 0, 0)
      x = tf.reshape(x, (-1, shape[axis] * shape[axis]))
      mask = K.not_equal(x, 0)
      x = tf.boolean_mask(x, mask)
      n = shape[axis]
      # print(n,(n * (n + 1)) // 2, (n * (n + 1)) // 2, '--------------')
      n = (n * (n + 1)) // 2 if self.interaction_itself else (n * (n - 1)) // 2
      # x.set_shape((shape[0], n))
      # x -= 1e10
      x = tf.reshape(x, (-1, n))
    return x

  def get_config(self):
    config = {
              'interaction_itself': self.interaction_itself, 
              'remove_duplicate': self.remove_duplicate, 
              'sqrt_norm': self.sqrtn_norm,
             }
    base_config = super(DotPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class DotInteractionPooling(Layer):
  def __init__(self, interaction_itself=False, remove_duplicate=False, l2_norm=False, **kwargs):
    super(DotInteractionPooling, self).__init__(**kwargs)
    self._self_interaction = interaction_itself
    self._skip_gather = not remove_duplicate
    self._l2_norm = l2_norm

  def call(self, x, sequence_length=None):
    if self._l2_norm:
      x = tf.nn.l2_normalize(x, axis=-1)
    batch_size = melt.get_shape(x, 0)
    num_features = melt.get_shape(x, 1)
    concat_features = x
    # https://github.com/tensorflow/recommenders/blob/v0.5.2/tensorflow_recommenders/layers/feature_interaction/dot_interaction.py#L22-L104
    # Interact features, select lower-triangular portion, and re-shape.
    xactions = tf.matmul(concat_features, concat_features, transpose_b=True)
    self.xactions = xactions
    gezi.set('dot_mat', xactions)
    ones = tf.ones_like(xactions)
    if self._self_interaction:
      # Selecting lower-triangular portion including the diagonal.
      lower_tri_mask = tf.linalg.band_part(ones, -1, 0)
      upper_tri_mask = ones - lower_tri_mask
      out_dim = num_features * (num_features + 1) // 2
    else:
      # Selecting lower-triangular portion not included the diagonal.
      upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
      lower_tri_mask = ones - upper_tri_mask
      out_dim = num_features * (num_features - 1) // 2

    if self._skip_gather:
      # Setting upper tiangle part of the interaction matrix to zeros.
      activations = tf.where(condition=tf.cast(upper_tri_mask, tf.bool),
                             x=tf.zeros_like(xactions),
                             y=xactions)
      out_dim = num_features * num_features
    else:
      activations = tf.boolean_mask(xactions, lower_tri_mask)
    activations = tf.reshape(activations, (batch_size, out_dim))
    return activations

  def get_config(self):
    config = {
              '_self_interaction': self._self_interaction, 
              '_skip_gather': self._skip_gather, 
              '_l2_norm': self._l2_norm,
             }
    base_config = super(DotInteractionPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class Cross(Layer):
  def __init__(self, num_layers=1, **kwargs):
    super(Cross, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.cross = tfrs.layers.dcn.Cross()

  def call(self, x, sequence_length=None):
    x1 = self.cross(x)
    if self.num_layers > 1:
      for i in range(self.num_layers - 1):
        x0 = x1
        x1 = self.cross(x, x1)
        x = x0
    return x1

class MultiHeadAttentionPooling(Layer):
  def __init__(self, num_heads, hidden_size=None, pooling='sum', return_att=False):
    super(MultiHeadAttentionPooling, self).__init__()
    self.num_heads = num_heads
    d_model = hidden_size
    self.d_model = d_model
    
    if d_model:
      assert d_model % self.num_heads == 0
      self.depth = d_model // self.num_heads
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)

    self.pooling = Pooling(pooling)
    self.return_att = return_att

  def build(self, input_shape):
    if not self.d_model:
      d_model = input_shape[-1]
      assert d_model % self.num_heads == 0, f'{d_model} {self.num_heads}'
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)
      self.depth = d_model // self.num_heads
      self.d_model = d_model
        
  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, x, sequence_length=None, axis=1):
    q, k, v = x, x, x
      
    mask = None if sequence_length is None else (1. - tf.sequence_mask(sequence_length, dtype=x.dtype))
    if mask is not None:
      mask = tf.cast(mask, x.dtype)
      mask = mask[:, tf.newaxis, tf.newaxis, :]

    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = melt.scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    
    output = self.pooling(output, sequence_length)
    if self.return_att:
      return output, attention_weights
    else:
      return output

SelfAttentionPooling = MultiHeadAttentionPooling

class FMPooling(Layer):
  def __init__(self,  
               **kwargs):
    super(FMPooling, self).__init__(**kwargs)
  
  def call(self, outputs, sequence_length=None, axis=1):
    summed_emb = melt.sum_pooling(outputs, sequence_length, axis)
    summed_emb_square = tf.square(summed_emb) 

    squared_emb = tf.square(outputs) 
    squared_sum_emb = melt.sum_pooling(squared_emb, sequence_length, axis)

    # [None * K]
    y_second_order = 0.5 * (summed_emb_square - squared_sum_emb) 
    return y_second_order

class Autoint(Layer):
  def __init__(self,  
               embedding_size=128,
               num_heads=2,
               num_layers=3,
               **kwargs):
    super(Autoint, self).__init__(**kwargs)
    from deepctr.layers.interaction import InteractingLayer
    self.att_layer_num = num_layers
    att_embedding_size = int(embedding_size / num_heads)
    att_head_num = num_heads
    att_res = True
    self.interact = InteractingLayer(
            att_embedding_size, att_head_num, att_res)

  def call(self, outputs, sequence_length=None, axis=1):  
    att_input = outputs
    for i in range(self.att_layer_num):
      # print(i, att_input.shape)
      att_input = self.interact(att_input)
    return att_input

class AutointPooling(Layer):
  def __init__(self,  
               embedding_size=128,
               num_heads=2,
               num_layers=3,
               **kwargs):
    super(AutointPooling, self).__init__(**kwargs)
    from deepctr.layers.interaction import InteractingLayer
    self.att_layer_num = num_layers
    att_embedding_size = int(embedding_size / num_heads)
    att_head_num = num_heads
    att_res = True
    self.interact = InteractingLayer(
            att_embedding_size, att_head_num, att_res)

  def call(self, outputs, sequence_length=None, axis=1):  
    att_input = outputs
    for i in range(self.att_layer_num):
      # print(i, att_input.shape)
      att_input = self.interact(att_input)
    # print(1, att_input.shape)
    att_output = tf.keras.layers.Flatten()(att_input)
    # print(2, att_output.shape)
    return att_output

class Autoint2Pooling(Layer):
  def __init__(self,  
               embedding_size=128,
               num_heads=2,
               num_layers=3,
               **kwargs):
    super(Autoint2Pooling, self).__init__(**kwargs)
    from deepctr.layers.interaction import InteractingLayer
    self.att_layer_num = num_layers
    att_res = True
    self.interact = InteractingLayer(
            int(embedding_size / num_heads), num_heads, att_res)
    self.pooling = Pooling('dot2')
  
  # TODO dynamic
  def build(self, input_shape):
    pass

  def call(self, outputs, sequence_length=None, axis=1):  
    att_input = outputs
    for i in range(self.att_layer_num):
      # print(i, att_input.shape)
      att_input = self.interact(att_input)
    # print(1, att_input.shape)
    # att_output = tf.keras.layers.Flatten()(att_input)
    att_output2 = self.pooling(att_input)
    return att_output2
    # att_output = tf.concat([att_output, att_output2], -1)
    # print(2, att_output.shape)
    # return att_output

#should be layer, Model will cause tf2 keras fail..
class Pooling(Layer):
  def __init__(self,  
               pooling,
               top_k=2,
               att_activation=tf.nn.relu,
               att_hidden=None,
               num_heads=4,
               **kwargs):
    super(Pooling, self).__init__(**kwargs)
    if gezi.get('activation') is not None:
      att_activation = gezi.get('activation')
    self.top_k = top_k

    name = pooling
    if not name:
      name = 'sum'

    pooling_str = pooling.replace(',', '_')
    self._name = f'{self.name}_{pooling_str}'

    self.poolings = []
    def get_pooling(name):
      if name == 'max':
        return MaxPooling()
      if name == 'sum':
        return SumPooling()
      elif name == 'mean' or name == 'avg':
        return MeanPooling()
      elif name == 'sqrtn' or name == 'sqrt' or name == 'sqrt_n':
        return SqrtnPooling()
      elif name == 'dense':
        return DensePooling()
      elif name == 'mlp':
        return MLPPooling()
      elif name == 'attention' or name == 'att':
        return AttentionPooling(att_hidden, activation=att_activation)
      elif name == 'att2': # att2 == latt
        return AttentionPooling(activation=att_activation, transform_inputs=False)
      elif name == 'att_2':
        return AttentionPooling(activation=att_activation, num_outputs=2)
      elif name == 'att_3':
        return AttentionPooling(activation=att_activation, num_outputs=3)
      elif name == 'att2_2':
        return AttentionPooling(activation=att_activation, num_outputs=2, transform_inputs=False)
      elif name == 'att2_3':
        return AttentionPooling(activation=att_activation, num_outputs=3, transform_inputs=False)
      elif name == 'att_dice':
        return AttentionPooling(att_hidden, activation=DiceActivation)
      elif name == 'linear_attention' or name == 'linear_att' or name == 'latt':
        return LinearAttentionPooling()
      elif name == 'nonlinear_attention' or name == 'nonlinear_att' or name == 'natt':
        return NonLinearAttentionPooling()
      elif name == 'topk' or name == 'top_k':
        return TopKPooling(top_k)
      elif name == 'top2':
        return TopKPooling(2)
      elif name == 'top3':
        return TopKPooling(3)
      elif name == 'top4':
        return TopKPooling(4)
      elif name == 'top5':
        return TopKPooling(5)
      elif name == 'top6':
        return TopKPooling(6)
      elif name == 'top7':
        return TopKPooling(7)
      elif name == 'topk_mean':
        return TopKMeanPooling(top_k)
      elif name == 'topk_weighted_mean':
        return TopKWeightedMeanPooling(top_k)
      elif name == 'topk_att':
        return TopKAttentionPooling(top_k)
      elif name =='first' or name == 'cls':
        return FirstPooling()
      elif name == 'last':
        return LastPooling()
      elif name == 'concat' or name == 'cat':
        return ConcatPooling()
      elif name == 'fm':
        return FMPooling()
      elif name == 'dot_pooling':
        return DotPooling()
      elif name == 'dot' or name == 'dot_int':
        # return DotPooling()
        # return tfrs.layers.feature_interaction.DotInteraction(self_interaction=False, skip_gather=False)
        return DotInteractionPooling()
      elif name == 'dot2':
        # return DotPooling(remove_duplicate=True)
        # return tfrs.layers.feature_interaction.DotInteraction(self_interaction=False, skip_gather=True)
        return DotInteractionPooling(remove_duplicate=True)
      elif name == 'dot3':
        return DotPooling(remove_duplicate=True)
      elif name == 'dot4':
        return DotPooling(remove_duplicate=True, sqrtn_norm=True)
      elif name == 'cosine2':
        return DotInteractionPooling(remove_duplicate=True, l2_norm=True)
      elif name == 'cosine3':
        return DotPooling(remove_duplicate=True, l2_norm=True)
      elif name == 'cin' or name =='xdeepfm':
        # Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, 
        # so try looking to see if a warning log message was printed above
        from deepctr.layers.interaction import CIN
        return CIN((128, 128,), 'relu', True, 0, 1024)
      elif name == 'din':
        return DinAttention(weight_normalization=False)
      elif name == 'din_norm':
        return DinAttention(weight_normalization=True)
      elif name == 'din_dice':
        return DinAttention(activation='dice', weight_normalization=False)
      elif name == 'din_norm_dice':
        return DinAttention(activation='dice', weight_normalization=True)
      elif name == 'mhead':
        return MultiHeadAttentionPooling(num_heads)
      elif name == 'mhead_dot':
        return MultiHeadAttentionPooling(num_heads, pooling='dot2')
      elif name == 'mhead_concat':
        return MultiHeadAttentionPooling(num_heads, pooling='concat')
      elif name == 'mhead_concat_dot':
        return MultiHeadAttentionPooling(num_heads, pooling='concat,dot2')
      elif name == 'autoint':
        return AutointPooling()
      elif name == 'autoint2':
        return Autoint2Pooling()
      elif name == 'sfu':
        return SFUPooling()
      else:
        raise ValueError('Unsupport pooling now:%s' % name)
        # return None

    self.names = name.split(',') if isinstance(name, str) else name
    for name in self.names:
      self.poolings.append(get_pooling(name))
    
    logging.debug('poolings:', self.poolings)
  
  def call(self, outputs, sequence_length=None, query=None, axis=1, calc_word_scores=False):
    results = []
    self.word_scores = []
    for i, pooling in enumerate(self.poolings):
      if query is None:
        results.append(pooling(outputs, sequence_length))
      else:
        try:
          results.append(pooling(outputs, sequence_length, query=query))
        except Exception:
          try:
            results.append(pooling(outputs, sequence_length))
          except Exception:
            results.append(pooling(outputs))
      if calc_word_scores:
        self.word_scores.append(melt.get_words_importance(outputs, sequence_length, top_k=self.top_k, method=self.names[i]))
    
    return tf.concat(results, -1)

  def get_config(self):
    config = {
              'top_k': self.top_k, 
              'poolings': self.poolings, 
              'names': self.names,
            }
    base_config = super(Pooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

Poolings = Pooling

class PoolingWrapper(Layer):
  def __init__(self,  
               pooling,
               **kwargs):
    super(PoolingWrapper, self).__init__(**kwargs)
    self.pooling = Pooling(pooling)

  def call(self, query, outputs, sequence_length):
    return self.pooling(outputs, sequence_length)

class DynamicDense(Layer):
  def __init__(self,  
               ratio,
               activation=None,
               use_bias=True,
               **kwargs):
    super(DynamicDense, self).__init__(**kwargs)
    self.ratio = ratio  
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.activation = activation
    self.use_bais = use_bias

  def build(self, input_shape):
    self.dense = layers.Dense(input_shape[-1] * self.ratio, activation_layer(self.activation), self.use_bais)

  def call(self, x):      
    return self.dense(x)

# https://tech.meituan.com/2018/03/29/recommend-dnn.html
class MissingValue(Layer):
  def __init__(self, output_dim, missing_value=0., **kwargs):
    super(MissingValue, self).__init__(**kwargs)
    self.dense = keras.layers.Dense(output_dim)
    self.missing_value = missing_value

  def call(self, x):
    is_missing = tf.cast(x < self.missing_value, x.dtype)
    return self.dense(x * (1. - is_missing)) + self.dense(is_missing)

class MultiDropout(Layer):
  def __init__(self,  
               output_dim=None,
               dims=[],
               drop_rate=0.3,
               num_experts=5,
               activation='relu',
               activate_last=False, #default like Dense not MLP
               **kwargs):
    super(MultiDropout, self).__init__(**kwargs)
    self.num_experts = num_experts
    if gezi.get('activation') is not None:
      activation_ = gezi.get('activation')
    if output_dim is not None:
      dims.append(output_dim)
    self.mlps = [MLP(dims, activation=activation_layer(activation), activate_last=activate_last) for _ in range(num_experts)] 
    self.dropouts = [keras.layers.Dropout(drop_rate) for _ in range(num_experts)] 

  def call(self, x):      
    xs = []
    for i in range(self.num_experts):
      x_i = self.dropouts[i](x)
      x_i = self.mlps[i](x_i)
      xs += [x_i]
    ret = tf.reduce_mean(tf.stack(xs, axis=1), axis=1)
    return ret

  def get_config(self):
    config = {
              'num_experts': self.num_experts, 
              'dense1': self.denses1, 
              'dense2': self.denses2,
              'dropouts': self.dropouts, 
              'activation': self.activation,
             }
    base_config = super(MultiDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

# TODO why Embedding not show when printing keras layers...
class Embedding(keras.layers.Layer):
  def __init__(self,  
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               trainable=True,
               num_shards=1,
               combiner=None,
               **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.initializer = embeddings_initializer
    self.input_dim, self.output_dim = input_dim, output_dim
    self.num_shards = num_shards

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)
    self.trainable = trainable

  def build(self, input_shape):
    num_shards = self.num_shards
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.input_dim, self.output_dim],
        dtype=tf.float32,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=self.trainable)
    self.built = True

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)

  def call(self, x, value=None, combiner=None):
    if x is not None:
      if isinstance(x, tf.Tensor):
        return tf.nn.embedding_lookup(self.embeddings, x)
      else:
        if isinstance(x, (list, tuple)):
          x, value = x[0], x[1]
        return tf.nn.embedding_lookup_sparse(self.embeddings, x, value, combiner=self.mode)
    else:
      return self.embeddings

class ModEmbedding(keras.layers.Layer):
  def __init__(self,  
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               trainable=True,
               input_length=None,
               num_shards=1,
               need_mod=True,
               l2_norm=False,
               num_buckets=None,
               combiner=None,
               mode='sum',
               append_weight=False,
               **kwargs):
    super(ModEmbedding, self).__init__(**kwargs)
    self.initializer = embeddings_initializer
    self.input_dim, self.output_dim = input_dim, output_dim
    self.num_shards = num_shards

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)
    self.need_mod = need_mod
    self.l2_norm = l2_norm
    self.mode = mode
    self.pooling = None
    self.trainable = trainable

  def build(self, input_shape):
    num_shards = self.num_shards
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.input_dim + 1, self.output_dim],
        dtype=tf.float32,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=self.trainable)
    if self.l2_norm:
      self.embeddings = tf.math.l2_normalize(self.embeddings)

    self.built = True

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)

  def call(self, x, value=None, combiner=None):
    if x is not None:
      if isinstance(x, tf.Tensor):
        mask = tf.cast(K.not_equal(x, 0), x.dtype)
        if self.need_mod:
          x = x % self.input_dim
          # x = tf.math.floormod(x, self.input_dim)
        x = (x + 1) * mask
        return tf.nn.embedding_lookup(self.embeddings, x)
      else:
        if isinstance(x, (list, tuple)):
          x, value = x[0], x[1]
        if self.need_mod:
          x = tf.sparse.SparseTensor(x.indices, x.values % self.input_dim, x.dense_shape)
        return tf.nn.embedding_lookup_sparse(self.embeddings, x, value, combiner=self.mode)
    else:
      return self.embeddings

# 不考虑hash 冲突的 QREmbedding 先确保正确 再考虑性能优化 也就是说 有一个固定的学习vector * 做输出
class MultiplyEmbedding(keras.layers.Layer):
  def __init__(self,  
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               trainable=True,
               input_length=None,
               num_shards=1,
               need_mod=False,
               split=False,
               num_buckets=None,
               combiner=None,
               mode='sum',
               append_weight=False,
               **kwargs):
    super(MultiplyEmbedding, self).__init__(**kwargs)
    self.initializer = embeddings_initializer
    self.input_dim, self.output_dim = input_dim, output_dim
    self.num_shards = num_shards

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)
    self.need_mod = need_mod
    self.mode = mode
    self.pooling = None
    self.trainable = trainable

  def build(self, input_shape):
    num_shards = self.num_shards
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.input_dim + 1, self.output_dim] if self.need_mod else [self.input_dim, self.output_dim],
        dtype=tf.float32,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=self.trainable)

    self.embeddings2 = self.add_weight(
        "embeddings2",
        shape=[1, self.output_dim],
        dtype=tf.float32,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer='uniform',
        regularizer=self.embeddings_regularizer,
        trainable=True)
    self.built = True

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)

  def call(self, x, value=None, combiner=None):
    if x is not None:
      if isinstance(x, tf.Tensor):
        if self.need_mod:
          mask = tf.cast(K.not_equal(x, 0), x.dtype)
          x = x % self.input_dim
          x = (x + 1) * mask
        return tf.nn.embedding_lookup(self.embeddings, x) * tf.nn.embedding_lookup(self.embeddings2, tf.zeros_like(x))
      else:
        if isinstance(x, (list, tuple)):
          x, value = x[0], x[1]
        if self.need_mod:
          x = tf.sparse.SparseTensor(x.indices, x.values % self.input_dim, x.dense_shape)
        return tf.nn.embedding_lookup_sparse(self.embeddings, x, value, combiner=self.mode)
    else:
      return self.embeddings

MEembedding = MultiplyEmbedding

# Not ok, not to use
class VocabEmbedding(layers.Layer):
  """An Embedding layer with 2 parts(second part fixed, not trainable)"""
  
  def __init__(self,input_dim, output_dim, base_dim=None, train_size=None, 
               embeddings_initializer='uniform', 
               embedding=None, trainable=True, 
               freeze_size=None, 
               drop_rate=0.,
               bias_emb=False,
               scale_emb=False, 
               use_one_hot=False, **kwargs):
    super(VocabEmbedding, self).__init__(**kwargs)
    self.vocab_size = input_dim
    self.freeze_size = freeze_size
    if train_size:
      self.freeze_size = self.vocab_size - train_size
    self.output_dim = output_dim
    self.base_dim = base_dim
    self.trainable = trainable

    self.embeddings = embedding
    self.embeddings2 = None
    if embedding is not None:
      if type(embedding) is str:
        if os.path.exists(embedding):
          embedding = np.load(embedding)
        else:
          embedding = None
      self.embeddings = embedding

    self.bias_emb = bias_emb
    self.scale_emb = scale_emb
    self.initializer = embeddings_initializer
    self.use_one_hot = use_one_hot

    if not base_dim or base_dim == output_dim:
      # self.proj = tf.identity # slow
      self.proj = lambda x: x
    else:
      self.proj = keras.layers.Dense(output_dim)

    stop_gradient = False
    if tf.__version__ < '2' and trainable == False:
      stop_gradient = True

    if not stop_gradient:
      # self.stop_gradient = tf.identity
      self.stop_gradient = lambda x: x
    else:
      self.stop_gradient = tf.stop_gradient

    if drop_rate > 0.:
      self.dropout = keras.layers.Dropout(drop_rate)
    else:
      self.dropout = lambda x: x
      
  def build(self, input_shape):
    initializer = self.initializer
    # some optimizer must use embedding on cpu 
    #with tf.device("/cpu:0"):
    if self.embeddings is not None:
      initializer = tf.constant_initializer(self.embeddings)
      logging.info('emb init from numpy pretrain and trainable:', self.trainable)
    # else:
    #   if FLAGS.emb_init == 'uniform':
    #     init_width = 0.5 / self.embedding_dim
    #     logging.info('emb random_uniform init with width:', init_width)
    #     initializer = tf.random_uniform_initializer(-init_width, init_width)
    #   elif FLAGS.emb_init == 'normal' or FLAGS.emb_init == 'random':
    #     stddev = FLAGS.emb_stddev or self.embedding_dim ** -0.5
    #     logging.info('emb random_normal init with stddev:', stddev)
        # initializer = tf.random_normal_initializer(mean=0., stddev=stddev)

    self.embeddings = self.add_weight(
        "embeddings",
        shape=[int(self.vocab_size), self.base_dim or self.output_dim],
        dtype=tf.float32,
        initializer=initializer,
        trainable=self.trainable)

    if self.freeze_size and self.trainable:
      embeddings, embeddings2 = tf.split(self.embeddings, [self.vocab_size - self.freeze_size, self.freeze_size], 0)
      self.embeddings = tf.concat([embeddings, tf.stop_gradient(embeddings2)], 0)
    
    if self.bias_emb:
      self.embeddings2 = self.add_weight(
        "embeddings2",
        shape=[1, self.base_dim or self.output_dim],
        dtype=tf.float32,
        initializer='uniform',
        trainable=True)

    super(VocabEmbedding, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)

  def call(self, x):
    #print('---------', self.embedding)
    if not self.bias_emb:
      ret = tf.nn.embedding_lookup(self.embeddings, x)
    else:
      ret = tf.nn.embedding_lookup(self.embeddings, x) * tf.nn.embedding_lookup(self.embeddings2, tf.zeros_like(x))
    if self.scale_emb:
      ret *= self.embedding_dim ** 0.5
    ret = self.stop_gradient(self.proj(self.dropout(ret)))
    return ret
  
  # # https://github.com/tensorflow/models/blob/ea61bbf06c25068dd8f8e130668e36186187863b/official/nlp/modeling/layers/on_device_embedding.py#L26
  # def call(self, inputs):
  #   flat_inputs = tf.reshape(inputs, [-1])
  #   if self.use_one_hot:
  #     one_hot_data = tf.one_hot(
  #         flat_inputs, depth=self.vocab_size, dtype=self.embeddings.dtype)
  #     embeddings = tf.matmul(one_hot_data, self.embeddings)
  #   else:
  #     embeddings = tf.gather(self.embeddings, flat_inputs)
  #   embeddings = tf.reshape(
  #       embeddings,
  #       # Work around b/142213824: prefer concat to shape over a Python list.
  #       tf.concat([tf.shape(inputs), [self.embedding_dim]], axis=0))
  #   embeddings.set_shape(inputs.shape.as_list() + [self.embedding_dim])
  #   if not self.bias_emb:
  #     return embeddings
  #   else:
  #     return embeddings * tf.nn.embedding_lookup(self.embeddings2, tf.zeros_like(inputs))

VEmbedding = VocabEmbedding

class SimpleEmbedding(ModEmbedding):
  def __init__(self,  
               input_dim,
               output_dim,
               num_buckets=None,
               need_mod=True,
               l2_norm=False,
               **kwargs):
    super(SimpleEmbedding, self).__init__(num_buckets or input_dim, output_dim, need_mod=True, **kwargs)

class PrEmbedding(keras.layers.Layer):
  def __init__(self, input_dim, output_dim, base_dim=None, 
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               trainable=True,
               train_size=None,
               drop_rate=0.,
               mask_zero=False,
               **kwargs):
    super(PrEmbedding, self).__init__(**kwargs)
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.embeddings_initializer = embeddings_initializer 
    self.embeddings_regularizer = embeddings_regularizer
    self.trainable = trainable
    self.mask_zero = mask_zero

    self.base_dim = base_dim

    if not base_dim or base_dim == output_dim:
      # self.proj = tf.identity # slow
      self.proj = lambda x: x
    else:
      self.proj = Project(output_dim)

    self.drop_rate = drop_rate
    if drop_rate > 0.:
      self.dropout = keras.layers.Dropout(dropout_rate)
    else:
      self.dropout = lambda x: x

    self.train_size = train_size if train_size and train_size < input_dim and trainable else None

  def build(self, input_shape):
    try:
      dtype=tf.float32 if not FLAGS.fp16 else tf.float16
    except Exception:
      dtype = tf.float32
    if not self.train_size:
      if isinstance(self.embeddings_initializer, np.ndarray):
        self.embeddings_initializer = tf.constant_initializer(self.embeddings_initializer)
      self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.input_dim, self.base_dim or self.output_dim],
        dtype=dtype,
        initializer=self.embeddings_initializer,
        regularizer=self.embeddings_regularizer,
        trainable=self.trainable,
        )
    else:
      input_dim1, input_dim2 = self.train_size, self.input_dim - self.train_size
      if isinstance(self.embeddings_initializer, np.ndarray):
        ndarray1, ndarray2 = np.split(self.embeddings_initializer, [input_dim1], axis=0)
        embeddings_initializer1, embeddings_initializer2 = tf.constant_initializer(ndarray1), tf.constant_initializer(ndarray2)
      else:
        embeddings_initializer1, embeddings_initializer2 = self.embeddings_initializer, self.embeddings_initializer
      embeddings1 = self.add_weight(
        "embeddings1",
        shape=[input_dim1, self.base_dim or self.output_dim],
        dtype=dtype,
        initializer=embeddings_initializer1,
        regularizer=self.embeddings_regularizer,
        trainable=True)

      embeddings2 = self.add_weight(
        "embeddings2",
        shape=[input_dim2, self.base_dim or self.output_dim],
        dtype=dtype,
        initializer=embeddings_initializer2,
        regularizer=self.embeddings_regularizer,
        trainable=False)

      self.embeddings = tf.concat([embeddings1, embeddings2], 0)

    self.built = True

  def call(self, x):
    x = tf.nn.embedding_lookup(self.embeddings, x)
    ret = self.proj(self.dropout(x))
    return ret

  def get_config(self):
    config = {
              'input_dim': self.input_dim, 
              'output_dim': self.output_dim, 
              'embeddings_initializer': self.embeddings_initializer,
              'embeddings_regularizer': self.embeddings_regularizer, 
              'trainable': self.trainable, 
              'train_size': self.train_size, 
              'drop_rate': self.drop_rate,
             }
    base_config = super(PrEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class LookupArray(keras.layers.Layer):
  def __init__(self, nparray, need_mod=False, **kwargs):
    super(LookupArray, self).__init__(**kwargs)
    if isinstance(nparray, str):
      self.nparray = np.load(nparray)
    else:
      self.nparray = nparray
    self.input_dim = self.nparray.shape[0]
    self.need_mod = need_mod

  def build(self, input_shape):
    # tf1 eager 不能跑 gpu
    # with tf.device('/cpu:0'): 
    device = '/cpu:0' if tf.__version__ < '2' and tf.executing_eagerly() else None
    with melt.device(device):
      self.lookups = self.add_weight(
          "lookups",
          shape=self.nparray.shape,
          dtype=melt.npdtype2tfdtype(self.nparray.dtype, large=True),
          initializer=tf.constant_initializer(self.nparray),
          trainable=False)

    self.built =True

  def call(self, x):
    if self.need_mod: 
      x = x % self.input_dim
    return tf.nn.embedding_lookup(self.lookups, x)

  def get_config(self):
    config = {'nparray': self.nparray, 'need_mod': self.need_mod}
    base_config = super(LookupArray, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

# TODO hash embedding currently not support embedding lookup with [None,] must be [None, None]
class HashEmbedding(keras.layers.Layer):
  """
    https://github.com/YannDubs/Hash-Embeddings
    https://github.com/dsv77/hashembedding
    http://papers.nips.cc/paper/7078-hash-embeddings-for-efficient-word-representations.pdf
    prefer to use QREmbedding
  """
  def __init__(self,  
               input_dim,
               output_dim,
               num_buckets=None,
               num_hashes=2,
               combiner='sum',
               mode='sum',
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               num_shards=1,
               need_mod=True,
               split=False,
               append_weight=False,
               lookup_name='hash_embedding',
               **kwargs):
    super(HashEmbedding, self).__init__(**kwargs)
    self.initializer = embeddings_initializer
    # if combiner == 'concat':
    #   assert output_imd % num_hashes == 0
    #   output_dim /= num_hashes
    self.input_dim, self.output_dim = input_dim, output_dim
    assert num_buckets 
    self.num_buckets = num_buckets
    self.num_hashes = num_hashes
    self.num_shards = num_shards
    self.combiner = combiner

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)

    self.append_weight = append_weight
    self.lookup_name = lookup_name
  
  def build(self, input_shape):
    num_shards = self.num_shards
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.num_buckets + 1, self.output_dim],
        dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=True)

    # TODO share or not share 
    assert self.num_buckets < 2**31
    np.random.seed(1024)
    tab = (np.random.randint(0, 2**31, size=(self.input_dim, self.num_hashes)) % self.num_buckets) + 1
    tab[0][0], tab[0][1] = 0, 0
    # with tf.device('/cpu'):
    with tf.compat.v1.variable_scope(self.lookup_name, reuse=tf.compat.v1.AUTO_REUSE):
      self.lookups = tf.compat.v1.get_variable('lookups', shape=(self.input_dim, self.num_hashes), initializer=tf.constant_initializer(tab), trainable=False, dtype=tf.int32)
    
    self.hash_weights = self.add_weight(
        "hash_weights",
        shape=[self.input_dim, self.num_hashes],
        dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=True)
    
    self.num_buckets = tf.constant(self.num_buckets, dtype=tf.int64)

    self.built =True

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)
        
  def call(self, x, combiner=None):
    if x is not None:
      shape = tf.shape(x)
      x = x % self.input_dim
      # [input_dim, num_hashes], [batch_size, len] -> [batch_size, len, num_hashes]
      ids = tf.nn.embedding_lookup(self.lookups, x)
      
      # [input_dim, num_hashes], [batch_size, len] -> [batch_size, len, num_hashes]
      weights = tf.nn.embedding_lookup(self.hash_weights, x)

      # [num_buckets, out_dim], [batch_size, len, num_hashes] - > [batch_size, len, num_hashes, out_dim]
      embs = tf.nn.embedding_lookup(self.embeddings, ids)

      # -> [batch_size, len, num_hashes, 1]
      weights_ = tf.expand_dims(weights, -1)

      # -> [batch_size, len, num_hashes, out_dim]
      embs *= weights_

      combiner = combiner or self.combiner
      if combiner == 'sum':
        # -> [batch_size, len, out_dim]
        ret = tf.reduce_sum(embs, -2)
      elif combiner == 'mul':
        ret = tf.reduce_prod(embs, -2)
      elif combiner == 'concat':
        ret = tf.reshape(embs, [shape[0], shape[1], self.output_dim * self.num_hashes])
      else:
        raise ValueError(combiner)
      if self.append_weight:
        ret = tf.concat([ret, weights], -1)
      return ret
    else:
      return self.embeddings

class HashEmbeddingV2(keras.layers.Layer):
  """
    HashEmbeddingV2 the same as HashEmbedding except for using two different hash as first % -> input_dim,
    so has less conflict comparing to HashEmbedding but it is slower.. so not egnough reason to replace HashEmbedding as performance not improve much
    Prefere to use QREmbedding
  """
  def __init__(self,  
                input_dim,
                output_dim,
                num_buckets=None,
                num_hashes=2,
                combiner='sum',
                mode='sum',
                embeddings_initializer='uniform',
                embeddings_regularizer=None,
                activity_regularizer=None,
                embeddings_constraint=None,
                mask_zero=False,
                input_length=None,
                num_shards=1,
                need_mod=True,
                split=False,
                append_weight=False,
                lookup_name='hash_embedding_v2',
                **kwargs):
    super(HashEmbeddingV2, self).__init__(**kwargs)
    self.initializer = embeddings_initializer
    self.input_dim, self.output_dim = input_dim, output_dim
    assert num_buckets 
    self.num_buckets = num_buckets
    self.num_hashes = num_hashes
    assert self.num_hashes == 2, "TODO for more num hahes"
    self.num_shards = num_shards
    self.combiner = combiner

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)

    self.append_weight = append_weight
    self.lookup_name = lookup_name

    def _is_prime(x):
      for i in range(2, int(np.sqrt(x))):
        if x % i == 0:
          return False
      return True

    def _next_prime(x):
        x -= 1
        while not _is_prime(x):
            x -= 1
        return x

    def _next_nprimes(x, n):
      ys = []
      y = x
      for i in range(n):
        y = _next_prime(y)
        ys += [y]
      return ys

    self.dims = _next_nprimes(self.input_dim, self.num_hashes - 1)

  def build(self, input_shape):
    num_shards = self.num_shards
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.num_buckets + 1, self.output_dim],
        dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=True)
    
    assert self.num_buckets < 2**31
    np.random.seed(1024)
    tab = (np.random.randint(0, 2**31, size=(self.input_dim, self.num_hashes)) % self.num_buckets) + 1
    tab[0][0], tab[0][1] = 0,0
    
    # with tf.device('/cpu'):
    with tf.compat.v1.variable_scope(self.lookup_name, reuse=tf.compat.v1.AUTO_REUSE):
      self.lookups = tf.get_variable('lookups', shape=(self.input_dim, self.num_hashes), initializer=tf.constant_initializer(tab), trainable=False, dtype=tf.int32)
        
    self.hash_weights = self.add_weight(
      "hash_weights",
      shape=[self.input_dim, self.num_hashes],
      dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
      partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
      initializer=self.initializer,
      regularizer=self.embeddings_regularizer,
      trainable=True)
    
    self.num_buckets = tf.constant(self.num_buckets, dtype=tf.int64)

    self.built = True

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)
    
  def call(self, x, combiner=None):
    if x is not None:
      x1 = x % self.input_dim
      xs = [x % dim for dim in self.dims]
      indexes = [x1, *xs]
      
      # [input_dim, num_hashes], [batch_size, len] -> [batch_size, len, num_hashes]
      weights = tf.nn.embedding_lookup(self.hash_weights, x1)

      retvals = []
      for i in range(self.num_hashes):
        ids = tf.nn.embedding_lookup(self.lookups[:,i], indexes[i])
        # [batch_size, len, out_dim]
        embs = tf.nn.embedding_lookup(self.embeddings, ids)
        # equal to tf.expand_dims(weights[:,:,i], -1)
        retvals.append(embs * tf.slice(weights, [0,0,i], [-1,-1,1]))

      combiner = combiner or self.combiner
      if combiner == 'sum':
        ret = sum(retvals)
      elif combiner == 'concat':
        ret = tf.concat(retvals, -1)
      else:
        raise ValueError(combiner)
      if self.append_weight:
        ret = tf.concat([ret, weights], -1)
    else:
      return self.embeddings

class QREmbedding(keras.layers.Layer):
  """
    DRLM compositional embedding using quotinent reminder strategy
    https://github.com/facebookresearch/dlrm
    https://arxiv.org/abs/1909.02107 Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems
    if num_buckets == input_dim means simple embedding with % to index
  """
  def __init__(self,  
               input_dim,
               output_dim,
               num_buckets=None,
               num_hashes=2,
               combiner='mul',
               mode='sum',
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               num_shards=1,
               need_mod=True,
               l2_norm=False,
               append_weight=False,
               **kwargs):
    super(QREmbedding, self).__init__(**kwargs)
    if input_dim < num_buckets:
      num_bucekts = input_dim

    self.initializer = embeddings_initializer
    self.num_hashes = num_hashes
    # if combiner == 'concat':
    #   assert output_dim % num_hashes == 0, f'{output_dim} {num_hashes}'
    #   output_dim /= num_hashes
    self.input_dim, self.output_dim = input_dim, output_dim
    if not num_buckets:
      num_buckets = input_dim
    self.num_buckets = num_buckets
    self.num_hashes = num_hashes
    self.num_shards = num_shards
    self.combiner = combiner
    self.mode = mode
    self.need_mod = need_mod
    self.l2_norm = l2_norm

    self.num_quotients = -(-input_dim // num_buckets)

    # print(input_dim, num_bucekts, self.num_quotients)

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)
    self.pooling = Pooling(mode)
  
  def build(self, input_shape):
    num_shards = self.num_shards
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.num_buckets + 1, self.output_dim],
        dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
        partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
        initializer=self.initializer,
        regularizer=self.embeddings_regularizer,
        trainable=True)

    if self.l2_norm:
      self.embeddings = tf.math.l2_normalize(self.embeddings)

    if self.num_quotients > 1:
      self.embeddings2 = self.add_weight(
          "embeddings2",
          shape=[self.num_quotients + 1, self.output_dim],
          dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
          partitioner=tf.compat.v1.fixed_size_partitioner(num_shards, axis=0) if num_shards > 1 else None,
          initializer=self.initializer,
          regularizer=self.embeddings_regularizer,
          trainable=True)
    
    self.built =True

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)
        
  def call(self, x, value=None, combiner=None, pooling=False):
    if x is not None:
      if isinstance(x, tf.Tensor):
        mask = tf.cast(K.not_equal(x, 0), x.dtype)

        if self.need_mod:
          x = x % self.input_dim
          # x = tf.math.floormod(x, self.input_dim)

        if self.input_dim == self.num_buckets:
          return tf.nn.embedding_lookup(self.embeddings, x)
        
        # TODO why tf.cast to int32 will turn much slower *2 times..
        x_rem = (x % self.num_buckets + 1) * mask
        # x_rem = (tf.math.floormod(x, self.num_buckets) + 1) * mask
        # NOTICE do not use cast like tf.cast(x/self.num_buckets, tf.int32)
        x_quo = (x // self.num_buckets + 1) * mask
        embs = tf.nn.embedding_lookup(self.embeddings, x_rem)
        if self.num_quotients > 1:
          embs2 = tf.nn.embedding_lookup(self.embeddings2, x_quo)
        # if pooling:
        #   if value is not None:
        #     embs *= tf.expand_dims(value, -1)
        #     len_ = None if self.mode == 'sum' else melt.length(x)
        #   else:
        #     len_ = melt.length(x) 
        #   embs = self.pooling(embs, len_)
        #   embs2 = self.pooling(embs2, len_)
      else:
        if isinstance(x, (list, tuple)):
          x, value = x[0], x[1]
        if self.need_mod:
          x = tf.sparse.SparseTensor(x.indices, x.values % self.input_dim, x.dense_shape)
        x_rem = x.values % self.num_buckets
        x_quo = x.values // self.num_buckets
        x_rem = tf.sparse.SparseTensor(x.indices, x_rem, x.dense_shape)
        x_quo = tf.sparse.SparseTensor(x.indices, x_quo, x.dense_shape)
        embs = tf.nn.embedding_lookup_sparse(self.embeddings, x_rem, value, combiner=self.mode)
        embs2 = tf.nn.embedding_lookup_sparse(self.embeddings2, x_quo, value, combiner=self.mode)

      combiner = combiner or self.combiner
      if combiner == 'mul':
        embs = embs * embs2
      elif combiner == 'sum':
        # -> [batch_size, len, out_dim]
        embs = embs + embs2
      elif combiner == 'concat':
        embs = tf.concat([embs, embs2], -1)
      else:
        raise ValueError(combiner)
      if pooling:
        if value is not None:
          embs *= tf.expand_dims(value, -1)
          len_ = None if self.mode == 'sum' else melt.length(x)
        else:
          len_ = melt.length(x) 
        embs = self.pooling(embs, len_)
      return embs
    else:
      return self.embeddings

  def get_config(self):
    config = {
      'initializer': self.initializer,
      'num_hashes': self.num_hashes,
      'input_dim': self.input_dim, 
      'output_dim': self.output_dim,
      'num_buckets': self.num_buckets,
      'num_shards': self.num_shards,
      'combiner': self.combiner, 
      'mode': self.mode,
      'need_mod': self.need_mod,
      'l2_norm': self.l2_norm,
      'num_quotients': self.num_quotients,
      'embeddings_regularizer': self.embeddings_regularizer,
      'pooling': self.pooling, 
      }

    base_config = super(QREmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
# TODO QRRembedding has problem of onnx conversion... not sure why though saved model ok, QREmbedding2 using keras.Embedding directly
# Still not work for onnx different results
class QREmbedding2(keras.layers.Layer):
  """
    DRLM compositional embedding using quotinent reminder strategy
    https://github.com/facebookresearch/dlrm
    https://arxiv.org/abs/1909.02107 Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems
    if num_buckets == input_dim means simple embedding with % to index
  """
  def __init__(self,  
               input_dim,
               output_dim,
               num_buckets=None,
               num_hashes=2,
               combiner='mul',
               mode='sum',
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               num_shards=1,
               need_mod=True,
               l2_norm=False,
               append_weight=False,
               **kwargs):
    super(QREmbedding2, self).__init__(**kwargs)
    if input_dim < num_buckets:
      num_bucekts = input_dim

    self.initializer = embeddings_initializer
    self.num_hashes = num_hashes
    # if combiner == 'concat':
    #   assert output_dim % num_hashes == 0, f'{output_dim} {num_hashes}'
    #   output_dim /= num_hashes
    self.input_dim, self.output_dim = input_dim, output_dim
    if not num_buckets:
      num_buckets = input_dim
    self.num_buckets = num_buckets
    self.num_hashes = num_hashes
    self.num_shards = num_shards
    self.combiner = combiner
    self.mode = mode
    self.need_mod = need_mod
    self.l2_norm = l2_norm

    self.num_quotients = -(-input_dim // num_buckets)

    # print(input_dim, num_bucekts, self.num_quotients)

    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    # self.activity_regularizer = keras.regularizers.get(activity_regularizer)
    self.pooling = Pooling(mode)
    
    self.embeddings = keras.layers.Embedding(self.num_buckets + 1, self.output_dim, 
                                             embeddings_initializer=embeddings_initializer,
                                             embeddings_regularizer=embeddings_regularizer,
                                             name='embeddings')
    if self.num_quotients > 1:
     self.embeddings2 = keras.layers.Embedding(self.num_quotients + 1, self.output_dim,
                                               embeddings_initializer=embeddings_initializer,
                                               embeddings_regularizer=embeddings_regularizer,
                                               name='embeddings2')
    else:
      self.embeddings2 = None

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)
        
  def call(self, x, value=None, combiner=None, pooling=False):
    mask = tf.cast(K.not_equal(x, 0), x.dtype)

    if self.need_mod:
      x = x % self.input_dim

    if self.input_dim == self.num_buckets:
      return self.embeddings(x)
    
    # TODO why tf.cast to int32 will turn much slower *2 times..
    x_rem = (x % self.num_buckets + 1) * mask
    # NOTICE do not use cast like tf.cast(x/self.num_buckets, tf.int32)
    x_quo = (x // self.num_buckets + 1) * mask
    embs = self.embeddings(x_rem)
    if self.num_quotients > 1:
      embs2 = self.embeddings2(x_quo)

    combiner = combiner or self.combiner
    if combiner == 'mul':
      embs = embs * embs2
    elif combiner == 'sum':
      # -> [batch_size, len, out_dim]
      embs = embs + embs2
    elif combiner == 'concat':
      embs = tf.concat([embs, embs2], -1)
    else:
      raise ValueError(combiner)
    if pooling:
      if value is not None:
        embs *= tf.expand_dims(value, -1)
        len_ = None if self.mode == 'sum' else melt.length(x)
      else:
        len_ = melt.length(x) 
      embs = self.pooling(embs, len_)
    return embs

  def get_config(self):
    config = {
      'initializer': self.initializer,
      'num_hashes': self.num_hashes,
      'input_dim': self.input_dim, 
      'output_dim': self.output_dim,
      'num_buckets': self.num_buckets,
      'num_shards': self.num_shards,
      'combiner': self.combiner, 
      'mode': self.mode,
      'need_mod': self.need_mod,
      'l2_norm': self.l2_norm,
      'num_quotients': self.num_quotients,
      'embeddings_regularizer': self.embeddings_regularizer,
      'pooling': self.pooling, 
      'embeddings': self.embeddings, 
      'embeddings2': self.embeddings2
      }

    base_config = super(QREmbedding2, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class EmbeddingBags(keras.layers.Layer):
  def __init__(self, input_dim, output_dim, num_buckets, 
               combiner=None, Embedding=None,
               return_list=False, mode='sum', embeddings_regularizer=None, 
               **kwargs):
    super(EmbeddingBags, self).__init__(**kwargs)
    if Embedding is None:
      Embedding = QREmbedding
    self.Embedding = Embedding
    self.input_dim, self.output_dim = input_dim, output_dim
    self.num_buckets = num_buckets
    self.combiner = combiner
    self.kwargs = kwargs
    self.pooling = Pooling(mode) if mode else lambda x, y: tf.identity(x)
    self.mode = mode
    self.split = split
    self.return_list = return_list
    self.embeddings_regularizer = embeddings_regularizer

  # NOTICE might be called multiple times like deep/emb wide/emb but not matters much
  def build(self, input_shape):
    # need to modify self.keys so copy 
    self.keys = gezi.get_global('embedding_keys').copy()
    try:
      if FLAGS.masked_fields:
        masked_keys = FLAGS.masked_fields.split(',')
        mask_mode = FLAGS.mask_mode.replace('_', '-').split('-')[-1]
        if 'regex' in FLAGS.mask_mode:
          if mask_mode == 'excl':
            def _is_ok(x):
              for key in masked_keys:
                if re.search(key, x):
                  return False
              return True
          else:
            def _is_ok(x):
              for key in masked_keys:
                if re.search(key, x):
                  return True
              return False
          self.keys = [x for x in self.keys if _is_ok(x)]
        else:  
          if mask_mode == 'excl':
            self.keys = [x for x in self.keys if x not in masked_keys]
          else:
            self.keys = masked_keys
        logging.debug('Final used onehot fields is:', ','.join(self.keys), 'count:', len(self.keys))
      if FLAGS.max_fields:
        np.random.shuffle(self.keys)
        self.keys = self.keys[:FLAGS.max_fields]
      gezi.set_global('embedding_keys', list(self.keys))
      gezi.set_global('fields', list(self.keys))
    except Exception:
      logging.warning(traceback.format_exc())
      pass

    num_buckets_ = int(self.num_buckets / len(self.keys))
    
    if not self.split:
      embedding = self.Embedding(self.input_dim, self.output_dim, self.num_buckets,
                                 mode=self.mode, combiner=self.combiner, **self.kwargs)
      self.embedding = embedding
    else:
      key_to_idims = gezi.get_global('embedding_input_dims', {})
      key_to_odims = gezi.get_global('embedding_output_dims', {})
      self.embeddings = {}
      self.linears = {}
      embedding_infos = gezi.get_global('embedding_infos', {})
      for key in self.keys:
        embedding_info = embedding_infos.get(key, {})
        num_buckets = key_to_idims.get(key, num_buckets_)
        output_dim = key_to_odims.get(key, self.output_dim)
        Embedding = self.Embedding
        if 'type' in embedding_info:
          Embedding = getattr(melt.layers, embedding_info['type'])
        num_buckets = embedding_info.get('input_dim', num_buckets)
        if num_buckets <= 20000:
          Embedding = SimpleEmbedding
        output_dim = embedding_info.get('output_dim', output_dim)
        mode = embedding_info.get('pooling', self.mode)
        if self.output_dim == 1:
          output_dim = 1
        input_dim = self.input_dim
        # input_dim = num_buckets * 10
        
        kwargs = self.kwargs.copy()
        kwargs['name'] = f'embedding_{key}'
        self.embeddings[key] = Embedding(input_dim, output_dim, num_buckets,
                                         embeddings_regularizer=self.embeddings_regularizer,
                                         mode=mode, combiner=self.combiner, 
                                         **self.kwargs)
        # logging.debug(key, Embedding, num_buckets, output_dim, pooling)
        if not hasattr(self.embeddings[key], 'pooling'):
          self.embeddings[key].pooling = None

        if output_dim != self.output_dim:
          self.linears[key] = keras.layers.Dense(self.output_dim)
        else:
          self.linears[key] = tf.identity

  def call(self, x, value=None, key=None, pooling=True):
    if key is not None:
      return self.deal(key, x, value)
    
    if self.split:
      l = []
      for i, key in enumerate(self.keys):
        # gpu = i % 8
        # with tf.device(f'/gpu:{gpu}'):
        emb = self.deal(key, x, value, pooling=pooling)
        l.append(emb)
      if self.return_list:
        return l
      else:
        return tf.stack(l, 1)
    else:
      l = []
      inputs = []
      values = []
      input_lens = []
      real_lens = []
      for i, key in enumerate(self.keys):
        input_lens.append(tf.shape(x[key])[1])
        if value is not None:
          values.append(value[key])
        real_lens.append(None if self.mode == 'sum' else melt.length(x[key]))
        inputs.append(x[key])
      input = tf.concat(inputs, 1)
      if value is not None:
        value = tf.concat(values, 1)
      embs = self.embedding(input)
      if value is not None:
        embs *= tf.expand_dims(value, -1)
      l = tf.split(embs, input_lens, axis=1)
      l = [self.pooling(x, len_) for x, len_ in zip(l, real_lens)]
      # l = tf.map_fn(self.pooling, l)
      if self.return_list:
        return l
      else:
        return tf.stack(l, 1)

  def deal(self, key, x, value=None, pooling=True):
    if self.split:
      embedding = self.embeddings[key]
    else:
      embedding = self.embedding
    if isinstance(x[key], tf.Tensor):
      if pooling and hasattr(embedding, 'pooling') and embedding.pooling:
        emb = embedding(x[key], value[key], pooling=True)
      else:
        embs = embedding(x[key])
        if value is not None and key in value:
          embs *= tf.expand_dims(value[key], -1)
          len_ = None if self.mode == 'sum' else melt.length(x)
        else:
          len_ = melt.length(x[key]) 
        if pooling:
          emb = self.pooling(embs, len_)
    else:
      emb = embedding(x[key], value[key])
    if self.split:
      emb = self.linears[key](emb)
    return emb

  def get_embedding(key):
    if self.split:
      return self.embeddings[key]
    else:
      return self.embedding

  # def call(self, x, value=None):
  #   def _deal(key):
  #     embs = self.embeddings[key](x[key])
  #     if value is not None and key in value:
  #       embs *= tf.expand_dims(value[key], -1)
  #       len_ = None
  #     else:
  #       len_ = melt.length(x[key]) 
  #     emb = self.pooling(embs, len_)
  #     return emb
  #   return tf.map_fn(_deal, self.keys, dtype=tf.float32)

class MLP(Layer):
  def __init__(self,  
               dims,
               activation='relu',
               activations=None,
               drop_rate=None,
               drop_indexes=None,
               batch_norm=False,
               batch_norms=None,
               layer_norm=False,
               layer_norms=None,
               input_dim=None,
               activate_last=True,
               **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.dims = dims
    self.activate_last = activate_last

    self.denses = [None] * len(dims)
    self.drops = [None] * len(dims)

    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.activation = activation
    self.activations = activations
    if activation == 'leaky_relu' or activation == 'leaky':
      activation=partial(tf.nn.leaky_relu, alpha=0.01)
    elif activation == 'dice' or activation == 'prelu':
      self.activations = [None] * len(dims)
      for i in range(len(dims)):
        self.activations[i] = DiceActivation if activation == 'dice' else PreluActivation
    
    if batch_norms:
      self.batch_norms = batch_norms
    else:
      if not batch_norm:
        self.batch_norms = [None] * len(dims)
      else:
        self.batch_norms = [tf.keras.layers.BatchNormalization(axis=-1) for _ in range(len(dims))]

    if layer_norms:
      self.layer_norms = layer_norms
    else:
      if not layer_norm:
        self.layer_norms = [None] * len(dims)
      else:
        self.layer_norms = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(len(dims))]

    self.output_dim = dims[-1]
    
    def _get_dim(dims, i):
      if dims[i]:
        return dims[i]
      if i == 0:
        assert input_dim, dims
        return input_dim // 2
      return dims[i - 1] // 2

    for i in range(len(dims)):
      dim = _get_dim(dims, i)
      activation_ = activation_layer(activation) if self.activations is None else activation_layer(self.activations[i])
      if not activate_last and i == len(dims) - 1:
        activation_ = None
      self.denses[i] = layers.Dense(dim, activation=activation_)
      if drop_rate and (not drop_indexes or i in drop_indexes):
        self.drops[i] = layers.Dropout(drop_rate)

  def get_config(self):
    config = {"dims":self.dims, "activation":self.activation, "activations": self.activations, "activate_last": self.activate_last}
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape[:-1].concatenate(self.output_dim)
  
  def call(self, x):
    for i in range(len(self.denses)):
      x = self.denses[i](x)
      if self.batch_norms[i] is not None:
        x = self.batch_norms[i](x)
      if self.layer_norms[i] is not None:
        x = self.layer_norms[i](x)
      if self.drops[i] is not None:
        x = self.drops[i](x)
    return x

class Residual(Layer):
  def __init__(self, dim, num_layers=1, **kwargs):
    super(Residual, self).__init__(**kwargs)
    self.linears = [layers.Dense(dim) for _ in range(num_layers)]
    self.batch_norms = [layers.BatchNormalization() for _ in range(num_layers)]
    self.num_layers = num_layers

  def call(self, x):
    for i in range(self.num_layers):
      x = x + self.batch_norms[i](self.linears[i](x))
      x = tf.nn.relu(x)

    return x

class Dropout(keras.layers.Layer):
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout).__init__(self, **kwargs)
    self.rate = rate 
    self.noise_shape = noise_shape
    self.seed = seed
  
  def call(self, x):
    if not K.learning_phase() or self.rate <= 0.:
      return x
    else:
      scale = 1.
      shape = tf.shape(input=x)
      if mode == 'embedding':
        noise_shape = [shape[0], 1]
        scale = 1 - self.rate
      elif mode == 'recurrent' and len(x.get_shape().as_list()) == 3:
        noise_shape = [shape[0], 1, shape[-1]] 
      return tf.nn.dropout(x, 1 - (1 - self.rate), noise_shape=noise_shape) * scale

# TODO remove self.step, using build()
class Gate(Layer):
  def __init__(self,
               drop_rate=0.,
               **kwargs):
    super(Gate, self).__init__(**kwargs)
    self.keep_prob = keep_prob
    self.dropout = kears.layers.Dropout(drop_rate)
    self.step = -1

  # def build(self, input_shape):
  #   dim = input_shape[-1]
  #   self.dense = layers.Dense(dim, use_bias=False, activation=tf.nn.sigmoid)

  def call(self, x, y):
    self.step += 1
    #with tf.variable_scope(self.scope):
    res = tf.concat([x, y], axis=-1)
    if self.step == 0:
      self.dense = layers.Dense(melt.get_shape(res, -1), use_bias=False, activation=tf.nn.sigmoid)
    d_res = self.dropout(res)
    gate = self.dense(d_res)
    return res * gate

class SemanticFusion(Layer):
  def __init__(self,
               drop_rate=0.,
               **kwargs):
    super(SemanticFusion, self).__init__(**kwargs)
    self.dropout = keras.layers.Dropout(drop_rate)

  def build(self, input_shape):
    dim = input_shape[-1]
    self.composition_dense = layers.Dense(dim, use_bias=True, activation=tf.nn.tanh, name='compostion_dense')
    self.gate_dense = layers.Dense(dim, use_bias=True, activation=tf.nn.sigmoid, name='gate_dense')

  def call(self, x, fusions):
    assert len(fusions) > 0
    vectors = tf.concat([x] + fusions, axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
    dv = self.dropout(vectors)
    r = self.composition_dense(dv)
    g = self.gate_dense(dv)
    return g * r + (1 - g) * x    

class SemanticFusionCombine(Layer):
  def __init__(self,
                drop_rate=0.,
                **kwargs):
      super(SemanticFusionCombine, self).__init__(**kwargs)
      self.sfu = SemanticFusion(drop_rate=drop_rate)
      # self.step = -1

  def build(self, input_shape):
    self.dense = layers.Dense(input_shape[-1], activation=None, name='sfu_dense')

  def call(self, x, y):
    # self.step += 1
    if melt.get_shape(x, -1) != melt.get_shape(y, -1):
      # if self.step == 0:
      #   self.dense = layers.Dense(melt.get_shape(x, -1), activation=None, name='sfu_dense')
      y = self.dense(y)
    return self.sfu(x, [y, x * y, x - y])

class SemanticFusionCombines(Layer):
  def __init__(self,
                drop_rate=0.,
                **kwargs):
      super(SemanticFusionCombines, self).__init__(**kwargs)
      self.worker = SemanticFusionCombine(drop_rate, **kwargs)

  def call(self, xs):
    res = []
    for i in range(len(xs)):
      for j in range(len(xs)):
        if j > i:
          res += [self.worker(xs[i], xs[j])]
    return res
  
# TODO may be need input_keep_prob and output_keep_prob(combiner dropout)
# TODO change keep_prob to use dropout
# https://github.com/HKUST-KnowComp/R-Net/blob/master/func.py
class DotAttention(Layer):
  def __init__(self,
               hidden=None,
               keep_prob=1.,
               drop_rate=None,
               combiner='gate',
               activation='relu',
               **kwargs):
    super(DotAttention, self).__init__(**kwargs)
    self.hidden = hidden
    if drop_rate is not None:
      keep_prob = 1 - drop_rate
    self.keep_prob = keep_prob
    self.combiner = combiner
    self.hidden = hidden
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.activation = activation
    if hidden:
      self.inputs_dense = layers.Dense(hidden, use_bias=False, activation=activation_layer(activation), name='inputs_dense')
      self.memory_dense = layers.Dense(hidden, use_bias=False, activation=activation_layer(activation), name='memory_dense')
    self.step = -1

    if combiner == 'gate':
      self.combine = Gate(keep_prob=keep_prob)
    elif combiner == 'sfu':
      self.combine = SemanticFusionCombine(keep_prob=keep_prob)
    elif combiner == None:
      self.combine = None
    else:
      raise ValueError(combiner)

  def call(self, inputs, memory, mask, self_match=False):
    combiner = self.combiner
    self.step += 1
    if self.step == 0 and self.hidden is None:
      out_dim = melt.get_shape(inputs, -1)
      self.inputs_dense = layers.Dense(out_dim, use_bias=False, activation=activation_layer(self.activation), name='inputs_dense')
      self.memory_dense = layers.Dense(out_dim, use_bias=False, activation=activation_layer(self.activation), name='memory_dense')
      self.hidden = out_dim
    # DotAttention already convert to dot_attention
    #with tf.variable_scope(self.scope):
    # TODO... here has some problem might for self match dot attention as same inputs with different dropout...Try self_match == True and verify..
    # NOTICE self_match == False following HKUST rnet
    d_inputs = dropout(inputs, keep_prob=self.keep_prob, training=K.learning_phase())
    if not self_match:
      d_memory = dropout(memory, keep_prob=self.keep_prob, training=K.learning_phase())
    else:
      d_memory = d_inputs
    JX = tf.shape(input=inputs)[1]
    
    inputs_ = self.inputs_dense(d_inputs)
    if not self_match:
      memory_ = self.memory_dense(d_memory)
    else:
      memory_ = inputs_

    scores = tf.matmul(inputs_, tf.transpose(a=memory_, perm=[0, 2, 1])) / (self.hidden ** 0.5)
    
    if mask is not None:
      mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
      #print(inputs_.shape, memory_.shape, weights.shape, mask.shape)
      # (32, 318, 100) (32, 26, 100) (32, 318, 26) (32, 318, 26)
      scores = softmax_mask(scores, mask)
    
    alpha = tf.nn.softmax(scores)
    self.alpha = alpha
    # logits (32, 326, 326)  memory(32, 326, 200)
    outputs = tf.matmul(alpha, memory)
    
    if self.combine is not None:
      return self.combine(inputs, outputs)
    else:
      return outputs

DotAttentionMatch = DotAttention

class SimpleAttentionMatch(Layer):
  def __init__(self,
               drop_rate=0.,
               combiner=None,
               **kwargs):
    super(SimpleAttentionMatch, self).__init__(**kwargs)
    keep_prob = 1 - drop_rate
    self.keep_prob = keep_prob
    self.combiner = combiner

    if combiner == 'gate':
      self.combine = Gate(keep_prob=keep_prob)
    elif combiner == 'sfu':
      self.combine = SemanticFusionCombine(keep_prob=keep_prob)
    elif combiner == None:
      self.combine = None
    else:
      raise ValueError(combiner)

  def call(self, inputs, memory, mask, self_match=False):
    combiner = self.combiner
    
    d_inputs = dropout(inputs, keep_prob=self.keep_prob, training=K.learning_phase())
    if not self_match:
      d_memory = dropout(memory, keep_prob=self.keep_prob, training=K.learning_phase())
    else:
      d_memory = d_inputs
    JX = tf.shape(input=inputs)[1]

    scores = tf.matmul(inputs, tf.transpose(a=memory, perm=[0, 2, 1])) 
    
    if mask is not None:
      mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
      #print(inputs_.shape, memory_.shape, weights.shape, mask.shape)
      # (32, 318, 100) (32, 26, 100) (32, 318, 26) (32, 318, 26)
      scores = softmax_mask(scores, mask)
    
    alpha = tf.nn.softmax(scores)
    self.alpha = alpha
    # logits (32, 326, 326)  memory(32, 326, 200)
    outputs = tf.matmul(alpha, memory)
    
    if self.combine is not None:
      return self.combine(inputs, outputs)
    else:
      return outputs

# https://arxiv.org/pdf/1611.01603.pdf
# but worse result then rnet only cq att TODO FIXME bug?
class BiDAFAttention(Layer):
  def __init__(self,
               hidden,
               drop_rate=1.0,
               combiner='gate',
               activation='relu',
               **kwargs):
    super(BiDAFAttention, self).__init__(**kwargs)
    self.hidden = hidden
    keep_prob = 1 - drop_rate
    self.keep_prob = keep_prob
    self.combiner = combiner
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.inputs_dense = layers.Dense(hidden, use_bias=False, activation=activation_layer(activation), name='inputs_dense')
    self.memory_dense = layers.Dense(hidden, use_bias=False, activation=activation_layer(activation), name='memory_dense')

    if combiner == 'gate':
      self.combine = Gate(keep_prob=keep_prob)
    elif combiner == 'sfu':
      self.combine = SemanticFusionCombine(keep_prob=keep_prob)
    elif combiner == None:
      self.combine = None
    else:
      raise ValueError(combiner)

  def call(self, inputs, memory, inputs_mask, memory_mask):
    combiner = self.combiner
    # DotAttention already convert to dot_attention
    #with tf.variable_scope(self.scope):
    d_inputs = dropout(inputs, keep_prob=self.keep_prob, training=K.learning_phase())
    d_memory = dropout(memory, keep_prob=self.keep_prob, training=K.learning_phase())
    JX = tf.shape(input=inputs)[1]
    
    inputs_ = self.inputs_dense(d_inputs)
    memory_ = self.memory_dense(d_memory)

    # shared matrix for c2q and q2c attention
    scores = tf.matmul(inputs_, tf.transpose(a=memory_, perm=[0, 2, 1])) / (self.hidden ** 0.5)

    # c2q attention
    mask = memory_mask
    if mask is not None:
      mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
      scores = softmax_mask(scores, mask)

    alpha = tf.nn.softmax(scores)
    self.alpha = alpha
    c2q = tf.matmul(alpha, memory)

    # TODO check this with allennlp implementation since not good result here...
    # q2c attention
    # (batch_size, clen)
    logits = tf.reduce_max(input_tensor=scores, axis=-1) 
    mask = inputs_mask
    if mask is not None:
      logits = softmax_mask(logits, mask)
    alpha2 = tf.nn.softmax(logits)
    # inputs (batch_size, clen, dim), probs (batch_size, clen)
    q2c = tf.matmul(tf.expand_dims(alpha2, 1), inputs)
    # (batch_size, clen, dim)
    q2c = tf.tile(q2c, [1, JX, 1])

    outputs = tf.concat([c2q, q2c], -1)

    if self.combine is not None:
      return self.combine(inputs, outputs)
    else:
      return outputs

BiDAFAttentionMatch = BiDAFAttention

# copy from mreader pytorch code which has good effect on machine reading
# https://github.com/HKUST-KnowComp/MnemonicReader
class SeqAttnMatch(Layer):
  """Given sequences X and Y, match sequence Y to each element in X.
  * o_i = sum(alpha_j * y_j) for i in X
  * alpha_j = softmax(y_j * x_i)
  """
  def __init__(self, 
               drop_rate=0.,  
               combiner='gate',
               activation='relu',
               identity=False):
    super(SeqAttnMatch, self).__init__()
    keep_prob = 1 - drop_rate
    self.keep_prob = keep_prob
    self.identity = identity
    self.step = -1
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.activation = activation

    if combiner == 'gate':
      self.combine = Gate(keep_prob=keep_prob)
    elif combiner == 'sfu':
      self.combine = SemanticFusionCombine(keep_prob=keep_prob)
    elif combiner == None:
      self.combine = None
    else:
      raise ValueError(combiner)

  # mask is y_mask
  def call(self, x, y, mask=None):
    self.step += 1
    x_ = x

    x = dropout(x, keep_prob=self.keep_prob, training=K.learning_phase())
    y = dropout(y, keep_prob=self.keep_prob, training=K.learning_phase())

    if self.step == 0:
      if not self.identity:
        self.linear = layers.Dense(melt.get_shape(x, -1), activation=activation_layer(self.activation))
      else:
        self.linear = None
    
    # NOTICE shared linear!
    if self.linear is not None:
      x = self.linear(x)
      y = self.linear(y)

    scores = tf.matmul(x, tf.transpose(a=y, perm=[0, 2, 1])) 

    if mask is not None:
      JX = melt.get_shape(x, 1)
      mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
      scores = softmax_mask(scores, mask)

    alpha = tf.nn.softmax(scores)
    self.alpha = alpha

    y = tf.matmul(alpha, y)

    if self.combine is None:
      return y
    else:
      return self.combine(x_, y)

class SelfAttnMatch(Layer):
  """
  * o_i = sum(alpha_j * x_j) for i in X
  * alpha_j = softmax(x_j * x_i)
  """
  def __init__(self, 
                drop_rate=0.,  
                combiner=None,
                identity=False, 
                activation='relu',
                diag=True):
    super(SelfAttnMatch, self).__init__()
    keep_prob = 1 - drop_rate
    self.keep_prob = keep_prob
    self.identity = identity
    self.diag = diag
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.activation = activation

    if combiner == 'gate':
      self.combine = Gate(keep_prob=keep_prob)
    elif combiner == 'sfu':
      self.combine = SemanticFusionCombine(keep_prob=keep_prob)
    elif combiner == None:
      self.combine = None
    else:
      raise ValueError(combiner)
      
      if not identity:
          self.linear = nn.Linear(input_size, input_size)
      else:
          self.linear = None
      self.diag = diag

  def build(self, input_shape):
    if not self.identity:
      self.linear = layers.Dense(input_shape[-1], activation=activation_layer(self.activation))
    else:
      self.linear = lambda x: x

    self.built = True

  def call(self, x, seqlen=None, mask=None):
    if mask is None and seqlen is not None:
      mask = (1. - tf.sequence_mask(seqlen, x.shape[1], dtype=x.dtype))
      #mask = (1. - tf.sequence_mask(seqlen, dtype=tf.float32))
    #print(mask.shape, x.shape, seqlen)

    x_ = x
    x = dropout(x, keep_prob=self.keep_prob, training=K.learning_phase())
    x = self.linear(x)

    scores = tf.matmul(x, tf.transpose(a=x, perm=[0, 2, 1])) 

    #  x = tf.constant([[[1,2,3], [4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]], dtype=tf.float32) # shape=(2, 3, 3)
    #  z = tf.matrix_set_diag(x, tf.zeros([2, 3]))
    if not self.diag:
      # TODO better dim
      dim0 = melt.get_shape(scores, 0)
      dim1 = melt.get_shape(scores, 1)
      scores = tf.linalg.set_diag(scores, tf.zeros([dim0, dim1]))

    if mask is not None:
      JX = melt.get_shape(x, 1)
      mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
      scores = softmax_mask(scores, mask)

    alpha = tf.nn.softmax(scores)
    self.alpha = alpha

    x = tf.matmul(alpha, x)
    if self.combine is None:
      return x
    else:
      return self.combine(x_, x)

# from https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/model_dice.py
class DinAttention(Layer):
  def __init__(self, mlp_dims=(80, 40), activation='sigmoid', weight_normalization=True, **kwargs):
    super(DinAttention, self).__init__(**kwargs)
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.mlp = MLP([*mlp_dims, 1] if mlp_dims[-1] != 1 else mlp_dims, activation=activation, activate_last=False)
    self.weight_normalization = weight_normalization
    self.query_att = True

  def build(self, input_shape):
    HIDDEN = input_shape[-1]
    activation = 'relu'
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.dense = layers.Dense(HIDDEN, activation=activation_layer(activation), name='din_align')

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, keys, keys_len, query, context=None):
    '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
    '''
    assert query != None
    dtype = keys.dtype
    if isinstance(keys, list or tuple):
      query, keys, keys_len = keys
      keys = tf.cast(keys, dtype)
      query = tf.cast(query, dtype)

    T = melt.get_shape(keys, 1)
    HIDDEN = melt.get_shape(keys, 2)
    query = tf.tile(query, [1, T])
    query = tf.reshape(query, [-1, T, HIDDEN])
    if context is not None:
      query = tf.concat([query, context], axis=-1)
      query = self.dense(query)
    keys = tf.cast(keys, dtype)
    query = tf.cast(query, dtype)
    din_all = tf.concat([query, keys, query - keys, query * keys], axis=-1)
    outputs = self.mlp(din_all) # [B, T, 1]
    outputs = tf.reshape(outputs, [-1, 1, T]) # [B, 1, T]
    # Mask
    key_masks = tf.sequence_mask(keys_len, T)   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    if self.weight_normalization:
      # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
      paddings = tf.ones_like(outputs) * (-2 ** 15 + 1)
    else:
      paddings = tf.zeros_like(outputs)

    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
    outputs = tf.cast(outputs, dtype)
    # Scale
    outputs = outputs / (tf.cast(HIDDEN, dtype) ** 0.5)

    # Activation
    if self.weight_normalization:
      outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    keys = tf.cast(keys, dtype)
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]
    outputs = tf.squeeze(outputs, 1)

    return outputs

# libowei
class DINAttention(layers.Layer):
  def __init__(self, attention_hidden_units=None, attention_activation="sigmoid", 
               supports_masking=True, mode=0, weight_normalization=True, **kwargs):
    super(DINAttention, self).__init__(**kwargs)
    if attention_hidden_units is None:
      attention_hidden_units = "80,40,1"
    self.attention_hidden_units = attention_hidden_units.split(",")
    self.attention_activation = attention_activation
    self.supports_masking = supports_masking
    self.mode = mode
    self.weight_normalization = weight_normalization

    self.denses = [None] * len(self.attention_hidden_units)
    for i in range(len(self.attention_hidden_units)):
      activation = None if i == len(self.attention_hidden_units) - 1 else self.attention_activation
      self.denses[i] = layers.Dense(self.attention_hidden_units[i], activation=activation)

  def build(self, input_shape):
    super(DINAttention, self).build(input_shape)

  def call(self, x, mask=None):
    '''
    i_emb:     [Batch_size, Hidden_units]
    hist_emb:        [Batch_size, max_len, Hidden_units]
    hist_len: [Batch_size]
    '''
    assert len(x) == 3

    i_emb, hist_emb, hist_len = x[0], x[1], x[2]
    # print(i_emb.shape,hist_emb.shape,hist_len.shape)
    hidden_units = K.int_shape(hist_emb)[-1]
    max_len = tf.shape(hist_emb)[1]

    i_emb = tf.tile(i_emb, [1, max_len])  # (batch_size, max_len * hidden_units)
    i_emb = tf.reshape(i_emb, [-1, max_len, hidden_units])  # (batch_size, max_len, hidden_units)

    if self.mode==0:
      concat = K.concatenate([i_emb, hist_emb, i_emb - hist_emb, i_emb * hist_emb], axis=2)  # (batch_size, max_len, hidden_units * 4)
    elif self.mode==1:
      concat = K.concatenate([i_emb, hist_emb, i_emb * hist_emb], axis=2)

    for i in range(len(self.attention_hidden_units)):
      outputs = self.denses[i](concat)
      concat = outputs

    outputs = tf.reshape(outputs, [-1, 1, max_len])  # (batch_size, 1, max_len)

    if self.supports_masking:
      mask = tf.sequence_mask(hist_len, max_len)  # (batch_size, 1, max_len)
      mask = tf.reshape(mask, [-1, 1, max_len])
      # print(mask[0])
      padding = tf.ones_like(outputs) * (-1e12)
      outputs = tf.where(mask, outputs, padding)

    outputs = outputs / (hidden_units ** 0.5)

    if self.weight_normalization:
      outputs = K.softmax(outputs)

    # print(hist_emb[0])
    outputs = tf.matmul(outputs, hist_emb)  # batch_size, 1, hidden_units)
    # print(outputs[0])

    # outputs = tf.squeeze(outputs)  # (batch_size, hidden_units)
    outputs = tf.reshape(outputs, [-1, hidden_units])

    return outputs


# TODO melt.layers.KVPooling for (query, keys)
      
# https://github.com/openai/finetune-transformer-lm/blob/master/train.py
class LayerNorm(keras.layers.Layer):
  def __init__(self, 
               e=1e-5, 
               axis=[1]):
    super(LayerNorm, self).__init__()
    self.step = -1
    self.e, self.axis = e, axis

  def call(self, x):
    self.step += 1
    if self.step == 0:
      n_state = melt.get_shape(x, -1)
      self.g = self.add_weight(
          "g",
          shape=[n_state],
          initializer=tf.compat.v1.constant_initializer(1))
      self.b = self.add_weight(
          "b",
          shape=[n_state],
          initializer=tf.compat.v1.constant_initializer(1))
    e, axis = self.e, self.axis
    u = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
    s = tf.reduce_mean(input_tensor=tf.square(x-u), axis=axis, keepdims=True)
    x = (x - u) * tf.math.rsqrt(s + e)
    x = x * self.g + self.b
    return x

class MultiHeadAttention(Layer):
  def __init__(self, num_heads, hidden_size=None, pooling='att', return_att=False):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    d_model = hidden_size
    self.d_model = d_model
    
    if d_model:
      assert d_model % self.num_heads == 0
      self.depth = d_model // self.num_heads
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)

    self.pooling = Pooling(pooling)
    self.return_att = return_att
    self.query_att = True

  def build(self, input_shape):
    if not self.d_model:
      d_model = input_shape[-1]
      assert d_model % self.num_heads == 0, f'{d_model} {self.num_heads}'
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)
      self.depth = d_model // self.num_heads
      self.d_model = d_model
        
  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, query, keys, sequence_length=None, axis=1):
    q, k, v = query, keys, keys
      
    mask = None if sequence_length is None else (1. - tf.sequence_mask(sequence_length, dtype=query.dtype))
    if mask is not None:
      mask = tf.cast(mask, query.dtype)
      mask = mask[:, tf.newaxis, tf.newaxis, :]

    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = melt.scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    
    output = self.pooling(output, sequence_length)
    if self.return_att:
      return output, attention_weights
    else:
      return output

MultiHeadAttentionMatch = MultiHeadAttention

class MultiHeadSelfAttention(Layer):
  def __init__(self, num_heads, hidden_size=None, return_att=False):
    super(MultiHeadSelfAttention, self).__init__()
    self.num_heads = num_heads
    d_model = hidden_size
    self.d_model = d_model
    
    if d_model:
      assert d_model % self.num_heads == 0
      self.depth = d_model // self.num_heads
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)

    self.return_att = return_att
    self.query_att = True

  def build(self, input_shape):
    if not self.d_model:
      d_model = input_shape[-1]
      assert d_model % self.num_heads == 0, f'{d_model} {self.num_heads}'
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.dense = tf.keras.layers.Dense(d_model)
      self.depth = d_model // self.num_heads
      self.d_model = d_model
        
  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, keys, sequence_length=None, axis=1):
    q, k, v = keys, keys, keys
      
    seq_len = melt.get_shape(q, 1)
    mask = None if sequence_length is None else (1. - tf.sequence_mask(sequence_length, maxlen=seq_len, dtype=q.dtype))
    if mask is not None:
      mask = tf.cast(mask, q.dtype)
      mask = mask[:, tf.newaxis, tf.newaxis, :]

    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = melt.scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    
    if self.return_att:
      return output, attention_weights
    else:
      return output

class MMoE(Layer):
  def __init__(self, num_tasks, num_experts, dims=None, mlp=None, activation='relu', return_att=False):
    super(MMoE, self).__init__()
    self.num_tasks = num_tasks
    self.num_experts = num_experts
    if gezi.get('activation') is not None:
      activation = gezi.get('activation')
    self.mlps = [copy.deepcopy(mlp) for _ in range(num_experts)] if mlp is not None else [MLP(dims, activation) for _ in range(num_experts)]
    # self.gates = [layers.Dense(num_experts, activation='relu') for _ in range(num_tasks)]
    self.gates = [layers.Dense(num_experts, use_bias=False, activation='softmax') for _ in range(num_tasks)]
    # self.drop = layers.Dropout(0.1)
    self.return_att = return_att

  def call(self, x):
    # [bs, hidden] * n
    outputs = [mlp(x) for mlp in self.mlps]
    # [bs, n, hidden]
    outputs = tf.stack(outputs, 1)
    res = []
    atts = []
    for i in range(self.num_tasks):
      # [bs, n, 1]
      # logits = self.gates[i](x)
      # logits = self.drop(logits)
      # logits = tf.linalg.l2_normalize(logits)
      # probs = tf.math.softmax(logits)
      probs = self.gates[i](x)
      # try:
      #   print(i, probs)
      # except Exception:
      #   pass
      probs = tf.expand_dims(probs, -1)
      outputs_ = outputs * probs
      output = tf.reduce_sum(outputs_, 1)
      res.append(output)
      atts.append(probs)

    if self.return_att:
      return res, atts
    else:
      return res

# class AutomaticWeightedLoss(nn.Module):
#     """automatically weighted multi-task loss
#     Params：
#         num: int，the number of loss
#         x: multi-task loss
#     Examples：
#         loss1=1
#         loss2=2
#         awl = AutomaticWeightedLoss(2)
#         loss_sum = awl(loss1, loss2)
#     """
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

# if __name__ == '__main__':
#     awl = AutomaticWeightedLoss(2)
#     print(awl.parameters())

class UncertaintyLoss(Layer):
  def __init__(self, num_objs):
    super(UncertaintyLoss, self).__init__()
    self.num_objs = num_objs

  def build(self, input_shape):
    self.sigmas_sq = [None] * self.num_objs
    for i in range(self.num_objs):
      self.sigmas_sq[i] = self.add_weight(name=f'sigmas_{i}',
                                          shape=[],
                                          initializer=tf.initializers.random_uniform(minval=0.2, maxval=1),                         
                                          # initializer=tf.zeros_initializer,
                                          dtype=tf.float32 if not FLAGS.fp16 else tf.float16,
                                          trainable=True)

  def call(self, losses):
    loss = 0.
    for i in range(len(self.sigmas_sq)):
      weight = self.sigmas_sq[i] ** 2
      factor = 0.5 / weight
      loss += factor * losses[i] + tf.math.log(1. + weight)
    return loss

# https://www.kaggle.com/hidehisaarai1213/glret21-efficientnetb0-baseline-inference
class GeM(tf.keras.layers.Layer):
    def __init__(self, pool_size=8, init_norm=3.0, normalize=False, **kwargs):
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.normalize = normalize

        super(GeM, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'init_norm': self.init_norm,
            'normalize': self.normalize,
        })
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.p = self.add_weight(name='norms', shape=(feature_size,),
                                 initializer=tf.keras.initializers.constant(self.init_norm),
                                 trainable=True)
        super(GeM, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = tf.math.maximum(x, 1e-6)
        x = tf.pow(x, self.p)

        # x = tf.nn.avg_pool(x, self.pool_size, self.pool_size, 'VALID')
        x = tf.pow(x, 1.0 / self.p)

        if self.normalize:
            x = tf.nn.l2_normalize(x, 1)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])
