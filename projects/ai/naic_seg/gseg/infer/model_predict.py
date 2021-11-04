#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model_predict.py
#        \author   chenghuige  
#          \date   2020-10-26 23:56:20.648944
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
import cv2
# from PIL import Image, ImageEnhance 
import tensorflow as tf
from tqdm.auto import tqdm

DEBUG = False
TEST_RAND_SHAPE = False
TEST_OOM = False
RAND_SIZE = 2048 # 256 * 8

if TEST_RAND_SHAPE:
  print('RAND_SIZE =', RAND_SIZE)

MIN_IMAGE_SIZE = (256, 256)
IMAGE_SIZE_ = 256
# assert IMAGE_SIZE_ % 32 == 0
IMAGE_SIZE = (IMAGE_SIZE_, IMAGE_SIZE_)

PAD_SIZE_ = 0
# assert PAD_SIZE_ % 32 == 0
PAD_SIZE = (PAD_SIZE_, PAD_SIZE_)

print("IMAGE_SIZE", IMAGE_SIZE, "PAD_SIZE", PAD_SIZE)

BATCH_SIZE = 49 # 7 * 7
print('batch_size:', BATCH_SIZE)

THRE = 0.125
print('THRE', THRE)

TTA = False
print('TTA:', TTA)

# 策略是如果要扩大 就不做resize 内部切割 最后一个前移一下位置
# 如果是缩小 那没有办法 只能resize 之所以设置THRE允许缩小 
# 因为如果都不缩 固定tile 256 那么可能很多比如260的小图都需要变成4图切割 计算量大了太多
def _get_scale(x, tile_size, thre):
  rate = (x % tile_size) / tile_size
  # 缩小图片
  if rate < thre:
    scale = max(x // tile_size, 1)
  else:
    scale = 0
    # scale = -(-x // tile_size)
  return scale

def _predict(model, imgs):
  if not TTA:
    pred = model.predict_on_batch(imgs)
    # pred = np.zeros([*imgs.shape, 1])
  else:
    imgs_list = [imgs, tf.image.flip_left_right(imgs), tf.image.flip_up_down(imgs)]
    imgs = tf.concat(imgs_list, axis=0)
    res = np.split(model.predict(imgs), len(imgs_list))
    res[1] = tf.image.flip_left_right(res[1]).numpy()
    res[2] = tf.image.flip_up_down(res[2]).numpy()
    # pred = np.mean(res, axis=0)
    pred = (res[0] + res[1] + res[2]) / 3.

  return pred

# shapes = set()
# max_patches = 0
# num_shapes = 0

# from https://github.com/PaddlePaddle/PaddleX/blob/4a6c125339d7c05550d470a413a947967500fd44/paddlex/cv/models/deeplabv3p.py
def overlap_tile_predict(model, image, tile_size, pad_size, batch_size):
  height, width, channel = image.shape
  image_tile_list = list()

  num_classes = model.output.shape[-1]

  # print('imaage shape for tile predict', image.shape, 'num_output_classes', num_classes)
  # Padding along the left and right sides
  if pad_size[0] > 0:
    left_pad = cv2.flip(image[0:height, 0:pad_size[1], :], 1)
    right_pad = cv2.flip(image[0:height, -pad_size[1]:width, :], 1)
    padding_image = cv2.hconcat([left_pad, image])
    padding_image = cv2.hconcat([padding_image, right_pad])
  else:
    padding_image = image

  # Padding along the upper and lower sides
  padding_height, padding_width, _ = padding_image.shape
  if pad_size[0] > 0:
    upper_pad = cv2.flip(
        padding_image[0:pad_size[0], 0:padding_width, :], 0)
    lower_pad = cv2.flip(
        padding_image[-pad_size[0]:padding_height, 0:padding_width, :],
        0)
    padding_image = cv2.vconcat([upper_pad, padding_image])
    padding_image = cv2.vconcat([padding_image, lower_pad])

  # crop the padding image into tile pieces
  padding_height, padding_width, _ = padding_image.shape

  h_count = -(-height // tile_size[0])
  w_count = -(-width // tile_size[1])

  # print(h_count, w_count)

  for h_id in range(0, h_count):
    for w_id in range(0, w_count):
      left = w_id * tile_size[1] 
      upper = h_id * tile_size[0] 
      right = min(left + tile_size[1] + pad_size[1] * 2,
                  padding_width)
      lower = min(upper + tile_size[0] + pad_size[0] * 2,
                  padding_height)

      if lower - upper < tile_size[0]:
        upper = lower - tile_size[0]

      if right - left < tile_size[1]:
        left = right - tile_size[1]
      
      image_tile = padding_image[upper:lower, left:right, :]

      # print(h_id, w_id, image_tile.shape[:2], (upper, lower), (left, right))
      image_tile_list.append(image_tile)

  # predict
  score_map = np.zeros((height, width, num_classes), dtype=np.float32)
  # count_map = np.zeros((height, width), dtype=np.float32)

  num_tiles = len(image_tile_list)

  # for i in tqdm(range(0, num_tiles, batch_size), desc='predicts', ascii=True, leave=False):
  for i in range(0, num_tiles, batch_size):
    begin = i
    end = min(i + batch_size, num_tiles)
    imgs = np.asarray(image_tile_list[begin:end])
    res = _predict(model, imgs)
    for j in range(begin, end):
      h_id = j // w_count
      w_id = j % w_count
      left = w_id * tile_size[1]
      upper = h_id * tile_size[0]
      right = min((w_id + 1) * tile_size[1], width)
      lower = min((h_id + 1) * tile_size[0], height)

      if lower - upper < tile_size[0]:
        upper = lower - tile_size[0]

      if right - left < tile_size[1]:
        left = right - tile_size[1]      

      tile_score_map = res[j - begin]
      tile_upper = pad_size[0]
      tile_lower = tile_score_map.shape[0] - pad_size[0]
      tile_left = pad_size[1]
      tile_right = tile_score_map.shape[1] - pad_size[1]

      # print(i, j, (h_id, w_id), (upper, lower), (left, right))

      score_map[upper:lower, left:right, :] = \
          tile_score_map[tile_upper:tile_lower, tile_left:tile_right, :]
      # count_map[upper:lower, left:right] += 1.

  # score_map /= count_map[:,:,np.newaxis]

  return score_map, num_tiles

def predict(model, input_path, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  ori_shape = img.shape[:2]
  resize_shape = None
  num_tiles = 0

  tile_size = IMAGE_SIZE
  scale_h = _get_scale(ori_shape[0], tile_size[0], THRE)
  scale_w = _get_scale(ori_shape[1], tile_size[1], THRE)

  resize_h = max(scale_h * tile_size[0] if scale_h else ori_shape[0], tile_size[0])
  resize_w = max(scale_w  * tile_size[1] if scale_w else ori_shape[1], tile_size[1])
  resize_shape = (resize_h, resize_w)

  # print('------------2', resize_shape)

  if resize_shape != ori_shape:
    img = tf.image.resize(img, (resize_shape[0], resize_shape[1])).numpy()
    # img = tf.image.resize(img, (resize_shape[0], resize_shape[1]), method='nearest').numpy()
    # img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
  
  scale = scale_h * scale_w
  if scale == 1:
    pred = _predict(model, np.asarray([img]))
    pred = pred[0]
    tile_size = resize_shape
    num_tiles = 1
  else:
    pred, num_tiles = overlap_tile_predict(model, img, tile_size=tile_size, pad_size=PAD_SIZE, batch_size=BATCH_SIZE)
  
  if pred.shape[:2] != ori_shape:
    # pred = tf.image.resize(pred, (ori_shape[0], ori_shape[1])).numpy()
    pred = tf.image.resize(pred, (ori_shape[0], ori_shape[1]), method='nearest').numpy()
    # pred = cv2.resize(img, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)
  
  if DEBUG:
    shape_info = (ori_shape, resize_shape, resize_shape[0] / ori_shape[0], tile_size, resize_shape[0] / tile_size[0], num_tiles)
    print(shape_info)

  # pred = pred.argmax(-1)
  # mask = (pred < 4).astype(np.int32)
  # pred = (pred + 1) * mask + (pred + 3) * (1 - mask)
  
  pred = pred.squeeze(-1)
  pred = pred.astype(np.uint8)    

  # assert pred.max() >= 1
  # assert pred.min() <= 17
  # assert 5 not in set(pred.reshape(-1))
  # assert 6 not in set(pred.reshape(-1))

  index, _ = os.path.splitext(os.path.basename(input_path))
  cv2.imwrite(os.path.join(output_dir, f'{index}.png'), pred)
