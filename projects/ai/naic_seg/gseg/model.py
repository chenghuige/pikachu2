#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2020-10-03 10:08:14.401495
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import math

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Lambda, Dense, Multiply, Add

from gezi import logging
import melt as mt
from .config import  *
from .util import *
from .augment import pred_augment
from .dataset import resize, resize_images
from .loss import get_loss_fn
from . import models_factory as factory

class Model(mt.Model):
  def __init__(self, model, backbone, **kwargs):
    super(Model, self).__init__(**kwargs)
  
    self.model = model

    if FLAGS.multi_scale:
      # TODO 如果share单一模型比如effb4 如何共享结构同时两个不同输入? 感觉不行？
      if not FLAGS.multi_scale_share:
        self.model2 = factory.get_model('fast_scnn', [*FLAGS.image_size2, 3])
      if FLAGS.multi_scale_attn:
        if not FLAGS.multi_scale_attn_dynamic:
          self.scale_attn_conv = Conv2D(
            filters=1,
            kernel_size=(3,3),  
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            activation='sigmoid',
            name='scale_attn_conv')
        else:
          self.scale_attn_conv = None
        #   def sse_attn(x, x2):
        #     weight = Conv2D(K.int_shape(x)[3], (1, 1), padding="same", kernel_initializer="he_normal",
        #       activation='sigmoid', strides=(1, 1), name="scale_sse_attn_conv")(x)
        #     return x * weight + x2 * (1 - weight)
        #   self.sse_attn = sse_attn

    # TODO 参考nv复杂一点
    #  attn = nn.Sequential(
    #     nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
    #     Norm2d(bot_ch),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
    #     Norm2d(bot_ch),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(bot_ch, out_ch, kernel_size=out_ch, bias=False),
    #     nn.Sigmoid())

    if FLAGS.scale_size:
      self.model2 = self.model if not gezi.get('model2') else gezi.get('model2')
      self.attn = Conv2D(
            filters=1,
            kernel_size=(3,3),  
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            activation='sigmoid',
            name='scale_attn_conv')

    self.preprocess = mt.image.get_preprocessing(backbone, FLAGS.normalize_image)
    logging.info('preprocess:', self.preprocess)
    if FLAGS.work_mode != 'train':
      logging.info('use_tta:', FLAGS.tta, 'tta:', FLAGS.tta_fns, FLAGS.tta_weights)
    self.activation = tf.keras.activations.get(FLAGS.ensemble_activation)
    if is_classifier() or FLAGS.multi_rate:
      self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
      self.dense = tf.keras.layers.Dense(FLAGS.NUM_CLASSES)
    
    if FLAGS.dataset_loss:
      self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
      self.dense1 = tf.keras.layers.Dense(256, activation='swish', name='data_loss_dense1')
      self.dense2 = tf.keras.layers.Dense(1, name='data_loss_dense')

    if FLAGS.multi_rate or FLAGS.dataset_loss:
      self.backbone = gezi.get('backbone')

    seg_notop = gezi.get('seg_notop')
    if seg_notop is not None and FLAGS.seg_weights:
      self.seg_notop = seg_notop
      seg_weights = f'{FLAGS.seg_weights}/seg_notop.h5' if os.path.isdir(FLAGS.seg_weights) else FLAGS.seg_weights
      logging.info('Load seg_weights from', seg_weights)
      self.seg_notop.load_weights(seg_weights)

    if FLAGS.use_mlp:
      self.mlp = mt.layers.MLP(FLAGS.mlp_dims, dropout=FLAGS.dropout, activation=gezi.get('activation') or 'relu')
      self.dense = tf.keras.layers.Dense(FLAGS.NUM_CLASSES)

    self.augmenter = None
    if FLAGS.augment_policy == 'auto':
      self.augmenter = mt.image.AutoAugment()
    elif FLAGS.augment_policy == 'rand':
      self.augmenter = mt.image.RandAugment()

    self.grad_checkpoint = FLAGS.grad_checkpoint

    # self.input_ = None
    # if hasattr(self, 'get_model'):
    #   self.get_model()

  def deal(self, x):
    #x = tf.cast(x, tf.int32)
    #x = tf.tile(tf.image.rgb_to_grayscale(x), [1,1,1,3])
    x2 = x

    if self.augmenter is not None:
      x = self.augmenter.distort(x)

    if not FLAGS.use_mlp:
      x = self.model(x)
    else:
      x = self.seg_notop(x)
      x = self.mlp(x)
      x = self.dense(x)

    if isinstance(x, (list, tuple)):
      out = None
      if len(x) == 1:
        x = x[0]
      elif len(x) == 2:
        x, notop_out = x

      if out is not None:
        self.out = out

    # if FLAGS.multi_scale:
    #   # 由于失误这里最初并没有走。。 所以不是multi scale 只是相同scale 叠加fast_scnn 看起来有一些收益仍然
    #   x2 = resize_images(x2, FLAGS.image_size2)
    #   x2 = self.model2(x2) 
    #   x2 = resize_images(x2, FLAGS.image_size)

    #   if not FLAGS.multi_scale_attn:
    #     x = x + x2 * FLAGS.multi_scale_weight
    #   else:
    #     if self.scale_attn_conv is None:
    #       self.scale_attn_conv = Conv2D(K.int_shape(x)[3], (3, 3), padding="same", kernel_initializer="he_normal",
    #                                     activation='sigmoid', strides=(1, 1), name="scale_sse_attn_conv")
    #     weight = self.scale_attn_conv(notop_out)
    #     x = x * weight + x2 * (1 - weight) 

    if FLAGS.scale_size or FLAGS.multi_scale:
      x2 = resize_images(x2, [FLAGS.scale_size, FLAGS.scale_size])
      x2, back_out2, notop_out2 = self.model2(x2)
      notop_out2 = tf.image.resize(notop_out2, notop_out.shape[1:3])
      notop_out2 = tf.cast(notop_out2, x2.dtype)
      cat_out = tf.concat([notop_out, notop_out2], axis=-1)
      attn = self.attn(cat_out)
      x2 = tf.image.resize(x2, x.shape[1:3])
      x2 = tf.cast(x2, x.dtype)
      x = (1. - attn) * x + attn * x2
    
    if is_classifier():
      x = self.global_average_layer(x)
      x = self.dense(x)

    ## 最终放弃mrate 没有特别用处 20201210
    if FLAGS.multi_rate:
      x2 = self.global_average_layer(self.out)
      y2 = self.dense(x2)
      if FLAGS.fp16:
        y2 = tf.keras.layers.Activation('linear', dtype='float32')(y2)
      self.y2 = y2

      # TODO since is logits should be - ? 检查下有问题再恢复 CHECK 如果改成-10效果变差了 说明应该multi_rate只需要多个class 目标loss 不需要这里修正像素级别结果 注意如果重启老代码程序需要代码保持前面一致..
      # TODO 或者 干脆 + tf.minimum(self.y2, 0.) ? 
      # 上面似乎效果都不好 暂时 multi_rate_strategy == 1 有收益 TODO 验证的时候是否关闭下面操作类似 验证采用 FLAGS.multi_rate_strategy == 0？
      if FLAGS.multi_rate_strategy == 1:
        x *= tf.cast(tf.math.sigmoid(y2) > FLAGS.classifier_threshold, x.dtype)[:,tf.newaxis, tf.newaxis,:]
      elif FLAGS.multi_rate_strategy == 2:
        x *= (tf.cast(tf.math.sigmoid(y2) > FLAGS.classifier_threshold, x.dtype) + K.epsilon())[:,tf.newaxis, tf.newaxis,:]
      elif FLAGS.multi_rate_strategy == 3:
        # TODO: try strategy 3 if < 0 before should not move it to 0
        # eff4 FPN ok 但是 eff4 Unet 刚好 OOM
        x_ = x * tf.cast(x < 0, x.dtype)
        x *= tf.cast(tf.math.sigmoid(y2) > FLAGS.classifier_threshold, x.dtype)[:,tf.newaxis, tf.newaxis,:] * tf.cast(x > 0, x.dtype)
        x += x_
      elif FLAGS.multi_rate_strategy == 4:
        pass
      elif FLAGS.multi_rate_strategy == 5: # not good result
        x += tf.minimum(self.y2, 0.)[:,tf.newaxis, tf.newaxis,:]
        # x -= (tf.cast(tf.math.sigmoid(y2) <= FLAGS.classifier_threshold, tf.float32)[:,tf.newaxis, tf.newaxis,:] * 10.)
      else:
        pass
    
    if FLAGS.dataset_loss:
      x2 = self.global_average_layer(self.out)
      x2 = self.dense1(x2)
      self.y_src = self.dense2(x2)

    return x

  def custom_save(self):
    if hasattr(self, 'seg_notop') and self.seg_notop is not None:
      self.seg_notop.save_weights(f'{FLAGS.model_dir}/seg_notop.h5')

  def call(self, input, training=False):
    self.input_ = input
    image = input['image'] if isinstance(input, dict) else input
    image = resize_images(image)
    image = self.preprocess(image)
  
    # print(iamge[0,:,:,0])
    # print(input['mask'])
    # print(tf.reduce_max(input['mask']))

    if not FLAGS.tta or FLAGS.tta_use_original:
      x = self.deal(image)

    # TODO 加在外面做一个通用tta wrapper class ?
    if FLAGS.tta and not training:
      tta_images = pred_augment(image, FLAGS.tta_fns, FLAGS.tta_intersect)
      xs = [self.deal(x) for x in tta_images]
      if not is_classifier():
        xs = pred_augment(xs, FLAGS.tta_fns, FLAGS.tta_intersect, reverse=True)

      if FLAGS.tta_use_original:
        xs = [x, *xs]
      
      xs = [self.activation(x) for x in xs]
    
      reduce_fn = tf.reduce_mean

      if FLAGS.tta_weights:
        reduce_fn = tf.reduce_sum
        assert len(FLAGS.tta_weights) == len(FLAGS.tta_fns)
        tta_weights = gezi.l1norm([1., *FLAGS.tta_weights])
        xs = [xs[i] * tta_weights[i] for i in range(len(xs))]
        # 如果报错 因为dataset比如train没有这个域 原因就是输入少key
        # AssertionError: Could not compute output Tensor("tf_op_layer_Mean/Mean_1:0", shape=(None, 256, 256, 8), dtype=float32)

      x = reduce_fn(tf.stack(xs, axis=1), axis=1)

    if is_classifier():
      return x

    if not FLAGS.dynamic_out_image_size:
      if x.shape[1:3] != FLAGS.ori_image_size:
        x = resize(x, FLAGS.ori_image_size)
      y = tf.reshape(x, [-1, *FLAGS.ori_image_size, FLAGS.NUM_CLASSES])
    else:
      y = x
    
    # if FLAGS.multi_object:
    #   y = mt.prob2logit(tf.math.softmax(y, -1) * tf.math.softmax(self.y2[:, tf.newaxis, tf.newaxis, :], -1))

    if FLAGS.adjust_preds and not training:
      y = tf.nn.softmax(y)
      weights = tf.constant(FLAGS.class_weights, dtype=tf.float32)
      # weights = tf.constant(np.asarray([math.log(1/x) ** 0.05 for x in gezi.get('class_weights')]), dtype=tf.float32)
      y *= weights

    if FLAGS.fp16:
      y = tf.keras.layers.Activation('linear', dtype='float32')(y)

    return y

  def get_model(self):
    if not FLAGS.dynamic_image_size:
      img_input = Input(shape=(*FLAGS.ori_image_size, 3), name='image')    
    else:
      img_input = Input(shape=(None, None, 3), name='image')

    # inp = {'image': img_input}
    # out = self.call(inp)
    # model = keras.Model(inp, out, name=f'{FLAGS.model}_{FLAGS.backbone}')
    # out = mt.build_model_with_precision(gezi.get('precision_policy_name'), self.call, img_input)
    out = self.call(img_input)
    model = keras.Model(img_input, out, name=f'{FLAGS.model}_{FLAGS.backbone}')
    # model.summary()
    return model

  # def get_model2(self):
  #   img_input = Input(shape=(*FLAGS.ori_image_size, 3), name='image')    
  #   id_input = Input(shape=(1,), name='id') 
  #   inp = {'image': img_input, 'id': id_input}
  #   out = self.call(inp)
  #   return keras.Model(inp, out, name=f'{FLAGS.model}_{FLAGS.backbone}')

  def get_loss(self):
    # return self.loss_wrapper(get_loss_fn(self.input_, self))
    return self.loss_wrapper(get_loss_fn())


def get_model(model_name):
  if model_name == 'None':
    return mt.Model()

  model = factory.get_model(model_name)
  model = Model(model, FLAGS.backbone)
  # loss_fn = model.get_loss()
  ## 如果不是functional model似乎不能保存完整graph 到.h5 但是saved model可以 载入预测也ok
  ## 当前先尝试save h5 之后 saved model 所以如果不使用.get_model保存graph是产出saved_model
  if FLAGS.functional_model:
    # model = model.get_model2()
    model = model.get_model()

  return model
