#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   augument.py
#        \author   chenghuige  
#          \date   2020-10-03 12:31:58.983404
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import random

import tensorflow as tf
import tensorflow_addons as tfa
# from tf_image.core.colors import rgb_shift, channel_drop

from gezi import logging
import melt as mt
from .config import *

## TODO 通用augment迁移到 melt.image.augment 
## 修改需要tpu验证测试单独 特别tf.function可能有问题 尽量其他函数不加 主augment有

# https://github.com/matterport/Mask_RCNN/issues/230
def random_crop_resize(img, mask, crop_shape=(228, 228), resize_shape=None, aug_mask=True):
    height, width = crop_shape
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    img_shape = (img.shape[0], img.shape[1]) if not resize_shape else resize_shape
    mask_shape = mask.shape
    x = tf.random.uniform(shape=[], maxval=img.shape[1] - width, dtype=tf.int32)
    y = tf.random.uniform(shape=[], maxval=img.shape[0] - height, dtype=tf.int32)
    img = img[y:y+height, x:x+width]
    img = tf.image.resize(img, img_shape)
    if aug_mask:
      mask = mask[y:y+height, x:x+width]
      mask = tf.image.resize(mask, img_shape, antialias=False, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    img = tf.reshape(img, [*img_shape, 3])
    mask = tf.reshape(mask, mask_shape)
    return img, mask

# TODO 似乎tpu不能接受这里 @tf.function 连接失败
def apply2(func, image, mask, p=0.5, aug_mask=True):
  val = tf.random.uniform(())
  if val <= p:
    image = func(image)
    if aug_mask:
      mask = func(mask)
  
  return image, mask

def augment_rot_oneof(image, mask, p=0.75, aug_mask=True):
  
  def _aug(image, mask, aug_mask=True):
    p = tf.random.uniform(())
    if p <= (1 / 3.):
      image = tf.image.rot90(image, k=1)
      if aug_mask:
        mask = tf.image.rot90(mask, k=1)
    elif p <= (2 / 3.):
      image = tf.image.rot90(image, k=2)
      if aug_mask:
        mask = tf.image.rot90(mask, k=2)
    else:
      image = tf.image.rot90(image, k=3)
      if aug_mask:
        mask = tf.image.rot90(mask, k=3)

    return image, mask
  
  val = tf.random.uniform(())
  if val < p:
    image, mask = _aug(image, mask, aug_mask=aug_mask)
  return image, mask

# 加强空间aug 增加rot90
def augment_spatial(image, mask, aug_mask=True):
  image, mask = apply2(tf.image.flip_left_right, image, mask, aug_mask=aug_mask)
  image, mask = apply2(tf.image.flip_up_down, image, mask, aug_mask=aug_mask)
  rotate_rate = FLAGS.rotate_rate or 0.5
  image, mask = apply2(tf.image.rot90, image, mask, p=rotate_rate, aug_mask=aug_mask)

  return image, mask

def augment_spatial2(image, mask, p, aug_mask=True):
  image, mask = apply2(tf.image.flip_left_right, image, mask, aug_mask=aug_mask)
  image, mask = apply2(tf.image.flip_up_down, image, mask, aug_mask=aug_mask)
  image, mask = augment_rot_oneof(image, mask, p=p, aug_mask=aug_mask)
  return image, mask

def augment_color_oneof(image):

  def _aug(image):
    p = tf.random.uniform(())
    if p <= (1 / 3.):
      image = tf.image.random_saturation(image, 0.7, 1.3)
    elif p <= (2 / 3.):
      image = tf.image.random_contrast(image, 0.8, 1.2)
    else:
      image = tf.image.random_brightness(image, 0.1)
    return image
  
  color_rate = FLAGS.color_rate or 0.15
  return mt.Apply(_aug, color_rate)(image)

def augment_color_compose(image):

  def _aug(image):
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)

    return image

  color_rate = FLAGS.color_rate or 0.15
  return mt.Apply(_aug, color_rate)(image)

def augment_sharpness(image, sharpen_rate=0.1, blur_rate=0.1):
  if not sharpen_rate and not blur_rate:
    return image

  assert sharpen_rate + blur_rate < 1
  p = tf.random.uniform(())
  if p <= sharpen_rate:
    rate = tf.random.uniform((), 1, 2)
    image = tfa.image.sharpness(image, rate)
  elif blur_rate and p <= sharpen_rate + blur_rate:
    rate = tf.random.uniform((), 0.1, 1)
    image = tfa.image.sharpness(image, rate)
  return image

# https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu
# TODO 如果蒸馏 需要 也对软标签 pred做cutmix
# TODO 实验增加cutmix + scale缩放
def cutmix(image, mask, pred=None, p=1.0):
  # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
  # output - a batch of images with cutmix applied
  DIM = mt.get_shape(image, 1)
  AUG_BATCH  = mt.get_shape(image, 0)

  # imgs = []
  # for j in range(AUG_BATCH):
  #   prob = tf.random.uniform(())
  ## TODO FIXME 很奇怪 这里必须用tf.cond 不能用prob < p if else 。。。 否则stack的时候 
  ## inaccessibleTensorError: The tensor 'Tensor("cond/strided_slice:0", shape=(256, 256, 3), dtype=float32)' cannot be accessed here: it is defined in another function or code block. Use return values, 
  ## explicit Python locals or TensorFlow collections to access it. Defined in: FuncGraph(name=cond_true_7099, id=140047006505040); accessed from: FuncGraph(name=Dataset_map_Dataset.post_decode, id=140047007789200).
  #   # if prob < p:
  #   #   imgs.append(image[j])
  #   # else:
  #   #   imgs.append(image[0])

  #   img = tf.cond(
  #     tf.less(prob, p),
  #     lambda: image[j],
  #     lambda: image[0])

  #   imgs.append(img)
  
  # image = tf.stack(imgs)

  # return image, mask

  def _deal(image, j, k, xa, xb, ya, yb):
    one = image[j,ya:yb,0:xa,:]
    two = image[k,ya:yb,xa:xb,:]
    three = image[j,ya:yb,xb:DIM,:]
    middle = tf.concat([one,two,three],axis=1)
    img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
    # img = tf.reshape(img, (DIM, DIM, 3))
    return img

  imgs = []; masks = []
  if pred is not None:
    preds = []

  for j in range(AUG_BATCH):
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    prob = tf.random.uniform(())
    k = tf.random.uniform([], 0, AUG_BATCH, dtype=tf.int32)
    # CHOOSE RANDOM LOCATION
    x = tf.random.uniform([], 0, DIM, dtype=tf.int32)
    y = tf.random.uniform([], 0, DIM, dtype=tf.int32)
    b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0

    P = tf.cast(prob <= p, tf.int32)
    WIDTH = tf.cast(DIM * tf.math.sqrt(1-b), tf.int32) * P
    ya = tf.math.maximum(0, y-WIDTH//2)
    yb = tf.math.minimum(DIM, y+WIDTH//2)
    xa = tf.math.maximum(0, x-WIDTH//2)
    xb = tf.math.minimum(DIM, x+WIDTH//2)
    # MAKE CUTMIX IMAGE
    img = _deal(image, j, k, xa, xb, ya, yb)
    imgs.append(img)
    mask_ = _deal(mask, j, k, xa, xb, ya, yb)
    masks.append(mask_)

    if pred is not None:
      pred_ = _deal(pred, j, k, xa, xb, ya, yb)
      preds.append(pred_)

  image2 = tf.stack(imgs, axis=0)
  mask2 = tf.stack(masks, axis=0)
  # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
  image2 = tf.reshape(image2, (AUG_BATCH,DIM,DIM,3))
  mask2 = tf.reshape(mask2, (AUG_BATCH,DIM,DIM,1))

  if pred is not None:
    pred2 = tf.stack(preds, axis=0)
    pred2 = tf.reshape(pred2, (AUG_BATCH,DIM,DIM, FLAGS.NUM_CLASSES))
    return image2, mask2, pred2
  
  return image2, mask2

def scaled_cutmix(image, mask, pred=None, p=1.0):
  # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
  # output - a batch of images with cutmix applied
  DIM = mt.get_shape(image, 1)
  AUG_BATCH  = mt.get_shape(image, 0)

  def _deal(image, j, k, xa, xb, ya, yb, xa2, xb2, ya2, yb2, method='bilinear'):
    one = image[j,ya:yb,0:xa,:]
    two = image[k,ya2:yb2,xa2:xb2,:]
    two = mt.image.resize(two, [yb - ya, xb - xa], method=method)
    three = image[j,ya:yb,xb:DIM,:]
    middle = tf.concat([one,two,three],axis=1)
    img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
    # img = tf.reshape(img, (DIM, DIM, 3))
    return img

  imgs = []; masks = []
  if pred is not None:
    preds = []

  for j in range(AUG_BATCH):
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    prob = tf.random.uniform(())
    k = tf.random.uniform([], 0, AUG_BATCH, dtype=tf.int32)
    # CHOOSE RANDOM LOCATION
    x = tf.random.uniform([], 0, DIM, dtype=tf.int32)
    y = tf.random.uniform([], 0, DIM, dtype=tf.int32)
    b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0

    P = tf.cast(prob <= p, tf.int32)
    WIDTH = tf.cast(DIM * tf.math.sqrt(1-b), tf.int32) * P
    WIDTH = tf.math.maximum(WIDTH, 2)
    ya = tf.math.maximum(0, y-WIDTH//2)
    yb = tf.math.minimum(DIM, y+WIDTH//2)
    xa = tf.math.maximum(0, x-WIDTH//2)
    xb = tf.math.minimum(DIM, x+WIDTH//2)

    lo, hi = FLAGS.cutmix_range[0], FLAGS.cutmix_range[1]
    scale = tf.random.uniform([], lo, hi)
    WIDTH2 = tf.cast(tf.cast(WIDTH, tf.float32) * scale, tf.int32)
    WIDTH2 = tf.math.maximum(WIDTH2, 2)

    masking = tf.cast(tf.random.uniform([]) <= FLAGS.cutmix_scale_rate, tf.int32)
    WIDTH2 = WIDTH2 * masking + WIDTH * (1 - masking)

    ya2 = tf.math.maximum(0, y-WIDTH2//2)
    yb2 = tf.math.minimum(DIM, y+WIDTH2//2)
    xa2 = tf.math.maximum(0, x-WIDTH2//2)
    xb2 = tf.math.minimum(DIM, x+WIDTH2//2)    

    # MAKE CUTMIX IMAGE
    img = _deal(image, j, k, xa, xb, ya, yb, xa2, xb2, ya2, yb2)
    imgs.append(img)
    mask_ = _deal(mask, j, k, xa, xb, ya, yb, xa2, xb2, ya2, yb2, method='nearest')
    masks.append(mask_)

    if pred is not None:
      pred_ = _deal(pred, j, k, xa, xb, ya, yb, xa2, xb2, ya2, yb2)
      preds.append(pred_)

  image2 = tf.stack(imgs, axis=0)
  mask2 = tf.stack(masks, axis=0)
  # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
  image2 = tf.reshape(image2, (AUG_BATCH,DIM,DIM,3))
  mask2 = tf.reshape(mask2, (AUG_BATCH,DIM,DIM,1))

  if pred is not None:
    pred2 = tf.stack(preds, axis=0)
    pred2 = tf.reshape(pred2, (AUG_BATCH,DIM,DIM, FLAGS.NUM_CLASSES))
    return image2, mask2, pred2
  
  return image2, mask2

## TODO gaussian blur hue 等等变换
## 暂时这个也去掉tf.function colab tpu之前也能跑 terminal也可以但是 local环境jupter这里function会报错 诡异 TODO
# @tf.function
def augment(image, mask=None, aug_mask=True):
  '''
  image: [B, H, W]
  mask:[B, H, 1] 
  '''

  if mask is None:
    aug_mask = False
    mask = tf.zeros_like(image[:,:,0])

  # < 0 means tested not good and depreciated
  if FLAGS.augment_level == 0:
    image, mask = apply2(tf.image.flip_left_right, image, mask, p=FLAGS.hflip_rate, aug_mask=aug_mask)
    image, mask = apply2(tf.image.flip_up_down, image, mask, p=FLAGS.hflip_rate, aug_mask=aug_mask)
  elif FLAGS.augment_level == 1:
    image, mask = augment_spatial(image, mask, aug_mask)
  elif FLAGS.augment_level == 2:
    image, mask = augment_spatial(image, mask, aug_mask)
    image = augment_color_oneof(image)
  elif FLAGS.augment_level == 3:  # diff with 4 will not do sharpness
    image, mask = augment_spatial(image, mask, aug_mask)
    image = augment_color_compose(image)
  elif FLAGS.augment_level == 4:  # diff with 3, by default will do sharpness
    image, mask = augment_spatial(image, mask, aug_mask)
    image = augment_color_compose(image)
    FLAGS.sharpen_rate = FLAGS.sharpen_rate or 0.1
    FLAGS.blur_rate = FLAGS.blur_rate or 0.1
    image = augment_sharpness(image, FLAGS.sharpen_rate, FLAGS.blur_rate)
  elif FLAGS.augment_level == 5:
    image, mask = augment_spatial2(image, mask, 0.75, aug_mask)
    image = augment_color_compose(image)
    image = augment_sharpness(image, FLAGS.sharpen_rate, FLAGS.blur_rate)
  elif FLAGS.augment_level == 6:
    image, mask = augment_spatial2(image, mask, 0.75, aug_mask)
    image = augment_color_compose(image)
  elif FLAGS.augment_level == 7: # 当前最佳
    image, mask = augment_spatial2(image, mask, 0.5, aug_mask)
    image = augment_color_compose(image)
  elif FLAGS.augment_level == 8:  
    image, mask = augment_spatial2(image, mask, 0.5, aug_mask)
    image = augment_color_compose(image)
    FLAGS.sharpen_rate = FLAGS.sharpen_rate or 0.1
    FLAGS.blur_rate = FLAGS.blur_rate or 0.1
    image = augment_sharpness(image, FLAGS.sharpen_rate, FLAGS.blur_rate)
  elif FLAGS.augment_level == -4:
    FLAGS.rotate_rate = 0.5
    image, mask = augment_spatial(image, mask, aug_mask)
    FLAGS.color_rate = 1.
    image = augment_color_compose(image)      
  else:
    raise ValueError(FLAGS.augment_level)
    
  return image, mask

# 注意不要@tf.function 否则不能save graph.. TODO
# @tf.function
def pred_augment(image, tta_fns, intersect=False, reverse=False):
  # fns = [getattr(tf.image,  tta_fn) for tta_fn in tta_fns]
  def _getaug(tta_fn):
    # TODO sharp not work..
    if tta_fn.startswith('sharpness'):
      rate = float(tta_fn.split('_')[-1])
      return lambda x: tfa.image.sharpness(x, rate)

    if tta_fn.startswith('rot90'):
      k = 1
      if '_' in tta_fn:
        k = int(tta_fn.split('_')[-1])
      return lambda x: tf.image.rot90(x, k)

    if tta_fn.startswith('size'):
      size_ = int(tta_fn[len('size'):])
      return lambda x: tf.image.resize(x, [size_, size_])

    return getattr(tf.image, tta_fn)
  
  fns = []
  for tta_fn in tta_fns:
    if not '-' in tta_fn: 
      fns.append(_getaug(tta_fn))
    else:
      def fn(x):
        tta_fns_ = tta_fn.split('-')
        for tta_fn_ in tta_fns_:
          x = _getaug(tta_fn_)(x)
        return x
      fns.append(fn)

  if intersect:
    fns_ = []
    for fn in fns:
      fns__ = []
      for i in range(len(fns_)):
        fns__.append(fns_[i])
        fns__.append([*fns_[i], fn])
      fns__.append([fn])
      fns_ = fns__

    fns = []
    for i in range(len(fns_)):
      def fn(x):
        for j in range(len(fns_[i])):
          x = fns_[i][j](x)
        return x
      fns.append(fn)

  if reverse:
    assert len(tta_fns) == len(fns)
    for i in range(len(fns)):
      if tta_fns[i].startswith('sharpness'):
        fns[i] = lambda x: x
      elif tta_fns[i].startswith('rot90'):
        k = 1
        if '_' in tta_fns[i]:
          k = int(tta_fns[i].split('_')[-1])
        fns[i] = lambda x: tf.image.rot90(x, k=4-k)
      elif tta_fn.startswith('size'):
        fns[i] = lambda x: tf.image.resize(x, FLAGS.ori_image_size)

  return mt.image.pred_augment(image, fns)
