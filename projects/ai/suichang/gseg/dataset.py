#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2020-04-12 20:33:51.902319
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
from tensorflow.keras import backend as K
# import tensorflow_io as tfio
import numpy as np

import gezi
import melt as mt
#from projects.ai.naic2020_seg.src import util
#from projects.ai.naic2020_seg.src.config import *
from .config import *
from .augment import *
from .util import is_classifier, pixel2class_label
from melt.image import resize
# 下面三个tf.function好像有造成用tf.function tpu运行失败

# @tf.function
def decode_image(image_data):
  image = tf.image.decode_image(image_data, channels=3)
  ## decode_tiff not work on tpu
  # image = tfio.experimental.image.decode_tiff(image_data)
  image = image[:, :, :3]
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [*FLAGS.ori_image_size, 3])

  ## TPU 这里不行 不能放在batch操作之前执行 放在后面也有问题 只能model内部处理 dataset不能resize
  # if FLAGS.image_size != FLAGS.ori_image_size:
  #   image = resize(image, FLAGS.image_size)
  #   image = tf.reshape(image, [*FLAGS.image_size, 3])
  return image

def resize_images(images, image_size=None):
  # 比如模型训练是按照256 * 256 输入 内部 resize到 288 * 288训练的 
  # infer输入 dynamic_image_size 例如 512 * 512 直接外部对image 做 resize 内部不再处理即可 比如 512 * (288 / 256) 但是这样不方便ensemble 要求相同输入
  # if FLAGS.dynamic_image_size:
  #   # return images
  #   if FLAGS.image_size == FLAGS.ori_image_size:
  #     return images
  #   else:
  #     if FLAGS.dynamic_image_scale:
  #       scale = FLAGS.image_size[0] / FLAGS.ori_image_size[0]
  #       scaled_shape = [tf.cast(tf.cast(tf.shape(images)[1], tf.float32) * scale, tf.int32),
  #                       tf.cast(tf.cast(tf.shape(images)[2], tf.float32) * scale, tf.int32)]
  #       images = resize(images, scaled_shape)
  #     return images

  # TODO check 如果是自由大图输入模式 动态输入 是多大 输出就是多大 针对infer only
  # TODO 专门区分infer 模式?
  if K.learning_phase() == -1:
    return images

  if FLAGS.image_sizes and K.learning_phase():
    # 这个地方很难达到预期 tf这个地方并不灵活 比较混乱 是否run_eagerly 单卡多卡 结果不一样 除了单卡run_eagerly结果都不符合 不能多尺度效果
    # 感觉还是可以考虑通过cutmix数据增强来达到多尺度
    assert not gezi.get('tpu'), 'tpu not support dynamic multi image scales'
    image_sizes = tf.constant(FLAGS.image_sizes, dtype=tf.int32)
    index = tf.random.uniform(shape=(), minval=0, maxval=len(FLAGS.image_sizes), dtype=tf.int32)
    image_size = image_sizes[index]
    if image_size != images.shape[1]:
      images = resize(images, [image_size, image_size])
      last_dim = images.shape[-1]
      images = tf.reshape(images, [-1, image_size, image_size, last_dim])
  elif FLAGS.image_scale_range:
    scale = tf.random.uniform(shape=(), minval=FLAGS.image_scale_range[0], maxval=FLAGS.image_scale_range[1])
    w = tf.cast(tf.cast(FLAGS.ori_image_size[0] * scale, tf.int32) / 32, tf.int32) * 32
    images = resize(images, [w, w])
  else:
    image_size = image_size or FLAGS.image_size
    if image_size != images.shape[1:3]:
      images = resize(images, image_size)
      last_dim = images.shape[-1]
      images = tf.reshape(images, [-1, *image_size, last_dim])
  # tf.print(images.shape)
  return images

# @tf.function
def decode_labels(mask):
  mask = tf.image.decode_png(mask, dtype=tf.dtypes.uint8)
  mask = tf.reshape(mask, [*FLAGS.ori_image_size, 1])
  mask = tf.cast(mask, tf.int32)
  return mask

def decode_nir(mask):
  mask = tf.image.decode_png(mask, dtype=tf.dtypes.uint8)
  mask = tf.reshape(mask, [*FLAGS.ori_image_size, 1])
  mask = tf.cast(mask, tf.float32)
  return mask

def decode_pred(pred):
  pred = tf.io.decode_raw(pred, tf.uint8)  
  pred = tf.reshape(pred, [*FLAGS.ori_image_size, FLAGS.NUM_CLASSES])  
  pred = tf.cast(pred, tf.float32) / 255.
  if FLAGS.fp16 and FLAGS.dataset_fp16:
    pred = mt.fp16(pred)
  return pred

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)
    assert not FLAGS.batch_parse, 'image decode must before batch'

    # self.use_post_decode = False
    self.use_post_decode = True

    self.class_lookup = None
    if FLAGS.class_lookup_flat:
      self.class_lookup = tf.expand_dims(tf.constant(CLASS_LOOKUP_FLAT, dtype=tf.int32), -1)
      # self.use_post_decode = True

    # self.teacher = None
    # if FLAGS.teacher:
    #   teacher_path = FLAGS.teacher if not os.path.isdir(FLAGS.teacher) else os.path.join(FLAGS.teacher, 'model.h5')
    #   with gezi.Timer(f'Loading teacher: {teacher_path}', print_fn=logging.info, print_before=True):
    #     self.teacher = mt.load_model(teacher_path)
    #     self.teacher.trainable = False
    #   self.use_post_decode = True

  def use_aug(self):
    if self.subset == 'train':
      return FLAGS.aug_train_image
    else:
      return FLAGS.aug_pred_image

  ## post_decode方式验证gpu ok 不过对应image resize还是tpu跑一段时间出问题 所以还是只能放回model处理
  def post_decode(self, x, y):
    # if 'image' in x:
    #   x['image'] = resize_images(x['image'])
    # if is_classifier():
    #   y = pixel2class_label(y)

    # TODO FIXME 这种方式data v1 验证按照v2的class 是正确的 但是未知原因tpu上面 model.evaluate 第二次会报错断开连接 HACK 暂时设置 FLAGS.vie == FLAGS.num_epochs 解决
    if self.class_lookup is not None:
      y = tf.nn.embedding_lookup(self.class_lookup, tf.squeeze(y, -1))
    elif FLAGS.onehot_dataset: # much slower here so just to onehot in loss.py is better
      y = tf.one_hot(tf.cast(tf.squeeze(y, -1), tf.int32), FLAGS.NUM_CLASSES)

    # if self.teacher is not None:
    #   y = self.teacher(x)

    # TODO 为啥会有valid  对应没有image域？ 一个train  3个valid 打印出来 最后一个没有image域 可能是info dataset ？ 去掉了 string ？ TODO 分析一下原因
    # print(x, y, self.subset, 'image' in x)
    if self.subset == 'train':
      if FLAGS.mixup_rate:
        # 效果很差
        x['image'], y, prob = mixup_transform(x['image'], y)
        y = tf.split(y, 2, axis=-1)
        y = (y[0], y[1], prob)
      elif FLAGS.mosaic_rate:
        x['image'], y = mosaic_transform(x['image'], y)
      else:
        if FLAGS.cutmix_rate:
          cutmix_fn = cutmix if not FLAGS.cutmix_range else scaled_cutmix
          if not 'pred' in x:
            x['image'], y = cutmix_fn(x['image'], y, p=FLAGS.cutmix_rate)
          else:
            x['image'], y, x['pred'] = cutmix_fn(x['image'], y, x['pred'], p=FLAGS.cutmix_rate)

    return x, y

  # @tf.function
  def parse(self, example):
    self.auto_parse()
    f = self.parse_(serialized=example)

    f['image'] = decode_image(f['image'])

    if FLAGS.no_labels:
      f['mask'] = tf.zeros_like(f['image'][:,:,0])
    else:
      if self.subset == 'test':
        f['mask'] = tf.zeros_like(f['id'])
      else:
        f['mask'] = decode_labels(f['mask'])

    if FLAGS.use_nir:
      nir = decode_nir(f['nir'])
      f['image'] = tf.concat([f['image'], nir], -1)

    if self.use_aug():
      f['image'], f['mask'] = augment(f['image'], f['mask'], aug_mask=self.subset != 'test')

    # 只用FLAGS.fp16判断控制也可以效果类似
    if FLAGS.fp16 and FLAGS.dataset_fp16:
      f['image'] = mt.fp16(f['image'])
      f['mask'] = mt.fp16(f['mask'])

    mt.try_append_dim(f)

    if FLAGS.soft_bce:
      f['bins'] = f['bins'] / tf.reduce_sum(f['bins'], -1)

    if FLAGS.distill and not FLAGS.teacher:
      if 'pred' in f:
        f['pred'] = decode_pred(f['pred'])
      
    x = f
    y = f['mask']
    del f['mask']

    ## TODO how to make work here ?
    # if is_classifier():
    #   pixels = FLAGS.ori_image_size[0] * FLAGS.ori_image_size[1]
    #   print(y.shape)
    #   y = tf.math.bincount(tf.reshape(y, (-1, pixels)), maxlength=NUM_CLASSES, axis=-1)
    
    if is_classifier():
      y = tf.cast(f['bins'] > 0, tf.int32)

    # y = tf.cast(y, tf.float32)

    # self.features = f
    return x, y
    
