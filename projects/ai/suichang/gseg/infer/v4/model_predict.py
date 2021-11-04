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
import tensorflow as tf
from tqdm.auto import tqdm

DEBUG = False
TEST_RAND_SHAPE = True
TEST_OOM = False

# DEBUG = True
# TEST_RAND_SHAPE = True
# TEST_OOM = True

MIN_IMAGE_SIZE = (256, 256)

# IMAGE_SIZE_ = os.environ.get('IMAGE_SIZE')
# IMAGE_SIZE_ = int(IMAGE_SIZE_) if IMAGE_SIZE_ else 1600
# IMAGE_SIZE_ = 3008
# IMAGE_SIZE_ = 2496
# IMAGE_SIZE_ = 512
# IMAGE_SIZE_ = 480
# IMAGE_SIZE_ = 640
IMAGE_SIZE_ = 256
# IMAGE_SIZE_ = 288
# IMAGE_SIZE_ = 1024
# IMAGE_SIZE_ = 1600

assert IMAGE_SIZE_ % 32 == 0
IMAGE_SIZE = (IMAGE_SIZE_, IMAGE_SIZE_)

# PAD_SIZE_ = os.environ.get('PAD_SIZE')
# PAD_SIZE_ = int(PAD_SIZE_) if PAD_SIZE_ else 32

# PAD_SIZE_ = 32
PAD_SIZE_ = 0
assert PAD_SIZE_ % 32 == 0
PAD_SIZE = (PAD_SIZE_, PAD_SIZE_)

print("IMAGE_SIZE", IMAGE_SIZE, "PAD_SIZE", PAD_SIZE)

# bs = os.environ.get('BATCH_SIZE')
# try:
#   BATCH_SIZE = int(bs) if bs else 4
# except Exception as e:
#   print(e)
#   BATCH_SIZE = 1

BATCH_SIZE = 1
# BATCH_SIZE = 4
print('batch_size:', BATCH_SIZE)

# DYNAMIC_TILE_SIZE = True
DYNAMIC_TILE_SIZE = False

print('DYNAMIC_TILE_SIZE', DYNAMIC_TILE_SIZE)

NUM_CLASSES = 15
m = {}
for i in range(NUM_CLASSES):
  if i < 4:
    m[i] = i + 1
  else:
    m[i] = i + 3

tta = False
# tta = True
# if os.environ.get('TTA') == '1':
#   tta = True

print('tta:', tta)

def _predict(model, imgs):
  if not tta:
    pred = model.predict(imgs)
  else:
    imgs_list = [imgs, tf.image.flip_left_right(imgs), tf.image.flip_up_down(imgs)]
    imgs = tf.concat(imgs_list, axis=0)
    res = np.split(model.predict(imgs), len(imgs_list))
    res[1] = tf.image.flip_left_right(res[1]).numpy()
    res[2] = tf.image.flip_up_down(res[2]).numpy()
    # pred = np.mean(res, axis=0)
    pred = (res[0] + res[1] + res[2]) / 3.
#     res0 = model.predict(imgs)
#     res1 = tf.image.flip_left_right(model.predict(tf.image.flip_left_right(imgs))).numpy()
#     res2 = tf.image.flip_up_down(model.predict(tf.image.flip_up_down(imgs))).numpy()
#     pred = (res0 + res1 + res2) / 3.

#     imgs_list = [imgs, tf.image.flip_left_right(imgs)]
#     imgs = tf.concat(imgs_list, axis=0)
#     res = np.split(model.predict(imgs), len(imgs_list))
#     res[1] = tf.image.flip_left_right(res[1]).numpy()
#     pred = (res[0] + res[1]) / 2.
  return pred

# def get_size(height, width):
#   h = round((height / 32) + 0.001) * 32
#   w = round((width / 32) + 0.001) * 32
#   # h = (-(-height // 32)) * 32
#   # w = (-(-width // 32)) * 32
#   return (h, w)

# shapes = set()
# max_patches = 0
# num_shapes = 0

# from https://github.com/PaddlePaddle/PaddleX/blob/4a6c125339d7c05550d470a413a947967500fd44/paddlex/cv/models/deeplabv3p.py
def overlap_tile_predict(model, image, tile_size, pad_size, batch_size):
  """有重叠的大图切小图预测。
  Args:
      img_file(str|np.ndarray): 预测图像路径，或者是解码后的排列格式为（H, W, C）且类型为float32且为RGB格式的数组。
      tile_size(list|tuple): 滑动窗口的大小，该区域内用于拼接预测结果，格式为（H, W）。默认值为[512, 512]。
      pad_size(list|tuple): 滑动窗口向四周扩展的大小，扩展区域内不用于拼接预测结果，格式为（H, W）。默认值为[64，64]。
      batch_size(int)：对窗口进行批量预测时的批量大小。默认值为32
  Returns:
      dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
          像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
  """

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
#     import copy
#     padding_image = copy.deepcopy(image)

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

  # print(padding_image.shape)

  hws = []
  # print('------abc', height // tile_size[1] + 1,  width // tile_size[0] + 1)
  assert height % tile_size[0] == 0
  assert width % tile_size[1] == 0

  # print('hscale', height / tile_size[1], 'wscale', width / tile_size[0])

  for h_id in range(0, int(height / tile_size[0])):
    for w_id in range(0, int(width / tile_size[1])):
      hws.append([h_id, w_id])

  for h_id, w_id in tqdm(hws, desc='sub_image', ascii=True, leave=False):
    left = w_id * tile_size[1] 
    upper = h_id * tile_size[0] 
    right = min(left + tile_size[1] + pad_size[1] * 2,
                padding_width)
    lower = min(upper + tile_size[0] + pad_size[0] * 2,
                padding_height)
    image_tile = padding_image[upper:lower, left:right, :]
    # print(left, upper, right, lower, image_tile.shape)
    image_tile_list.append(image_tile)

  # predict
  score_map = np.zeros((height, width, num_classes), dtype=np.float32)

  # print('score map', score_map.shape)

  num_tiles = len(image_tile_list)
  for i in tqdm(range(0, num_tiles, batch_size), desc='predict', ascii=True, leave=False):
    begin = i
    end = min(i + batch_size, num_tiles)
    # print([x.shape for x in image_tile_list[begin:end]])
    imgs = np.asarray(image_tile_list[begin:end])
    # print(imgs.shape)
    res = _predict(model, imgs)
    # print(res.shape)
    for j in range(begin, end):
      h_id = j // int(width / tile_size[0])
      w_id = j % int(width / tile_size[1])
      left = w_id * tile_size[1]
      upper = h_id * tile_size[0]
      right = min((w_id + 1) * tile_size[1], width)
      lower = min((h_id + 1) * tile_size[0], height)
      tile_score_map = res[j - begin]
      tile_upper = pad_size[0]
      tile_lower = tile_score_map.shape[0] - pad_size[0]
      tile_left = pad_size[1]
      tile_right = tile_score_map.shape[1] - pad_size[1]
      score_map[upper:lower, left:right, :] = \
          tile_score_map[tile_upper:tile_lower, tile_left:tile_right, :]

  return score_map, num_tiles

def concat_tile_predict(model, img, tile_size, batch_size):
  img_shape = img.shape[:2]
  assert img_shape[0] % tile_size[0] == 0
  imgs = []
  i = 0
  hws = []
  for h in range(0, img_shape[0], tile_size[0]):
    for w in range(0, img_shape[1], tile_size[1]):
      hws.append((h,w))

  preds = []
  for h, w in tqdm(hws, desc='sub_image', ascii=True, leave=False):
    left = w 
    upper = h
    right = min(w + tile_size[1], img_shape[1])
    lower = min(h + tile_size[0], img_shape[0])
    tile_img = img[upper:lower, left:right, :]
    # print(i, h, w, left, upper, right, lower, tile_img.shape)
    imgs.append(tile_img)
    if len(imgs) == batch_size:
      pred = _predict(model, np.asarray(imgs))
      # print(pred.shape)
      preds.append(pred)
      imgs = []
    i += 1
  if imgs:
    pred = _predict(model, np.asarray(imgs))
    preds.append(pred)
    imgs = []

  preds = np.concatenate(preds, axis=0)
  assert len(preds) > 1
  i = 0
  l = []
  for _ in range(0, img_shape[0], tile_size[0]):
    l_ = []
    for _ in range(0, img_shape[1], tile_size[1]):
      l_.append(preds[i])
      i += 1
    l.append(cv2.hconcat(l_))
  pred = cv2.vconcat(l)
  num_tiles = len(preds)
    
  return pred, num_tiles

def predict(model, input_path, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if TEST_RAND_SHAPE:
      if not TEST_OOM:
        # x = np.random.randint(256, 6000)
        #x = np.random.randint(256, 2049)
        x = 500
      else:
        x = IMAGE_SIZE[0] * 4
      img = cv2.resize(img, (x, x), interpolation=cv2.INTER_NEAREST)

  ori_shape = img.shape[:2]
  # print('ori shape', ori_shape)
  resize_shape = None
  num_tiles = 0

  # resize_h, resize_w = get_size(ori_shape[0], ori_shape[1])
  # resize_shape = (resize_h, resize_w)

  # if resize_shape[0] * resize_shape[1] <= IMAGE_SIZE[0] * IMAGE_SIZE[1]:
  #   if resize_shape != ori_shape:
  #     img = tf.image.resize(img, (resize_shape[0], resize_shape[1])).numpy()
  #     # img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
    
  #   pred = _predict(model, np.asarray([img]))
  #   pred = pred[0]
  #   tile_size = resize_shape
  #   num_tiles = 1
  # else:
  if not DYNAMIC_TILE_SIZE or ori_shape[0] != ori_shape[1]:
    tile_size = IMAGE_SIZE
    scale_h = round(ori_shape[0] / tile_size[0] + 0.001)
    scale_w = round(ori_shape[1] / tile_size[1] + 0.001)

    # scale_h = (-(-ori_shape[0] // tile_size[0]))
    # scale_w = (-(-ori_shape[1] // tile_size[1]))
  else:
    # assume square image
    max_scale = 0
    min_rate = 1.
    tile_size = IMAGE_SIZE[0]
    scale = 1
    min_image_size = int(IMAGE_SIZE[0] * 0.5)
    for tile_size_ in reversed(range(MIN_IMAGE_SIZE[0], IMAGE_SIZE[0] + 1, 32)):
      scale_ =  round(ori_shape[0] / tile_size_ + 0.001)
      if not max_scale:
        max_scale = scale_
      
      if scale_ - max_scale > 1:
        break
      
      resize_ = scale_ * tile_size_
      if resize_ > ori_shape[0]:
        rate = (resize_ - ori_shape[0]) / ori_shape[0]
      else:
        rate = (ori_shape[0] - resize_) / ori_shape[0]
        rate *= 1.2
#         print(ori_shape[0], tile_size_, rate)
      if rate < min_rate:
        min_rate = rate
        tile_size = tile_size_
        scale = scale_
      
      if tile_size_ < min_image_size:
        break

    scale_h, scale_w = scale, scale
    tile_size = (tile_size, tile_size)

    # print(ori_shape[0], tile_size, min_rate, scale)
  
  resize_h = scale_h * tile_size[0]
  resize_w = scale_w  * tile_size[1]
  resize_shape = (resize_h, resize_w)

  if resize_shape != ori_shape:
    img = tf.image.resize(img, (resize_shape[0], resize_shape[1])).numpy()
    # img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)

  if scale_h * scale_w > 1:
    if tile_size[0] > 512:
      batch_size = BATCH_SIZE
    elif tile_size[0] > 256:
      batch_size = 2
    else:
      batch_size = 4
#       if PAD_SIZE[0] * PAD_SIZE[1] == 0:
#         pred, num_tiles = concat_tile_predict(model, img, tile_size=tile_size, batch_size=BATCH_SIZE)
#       else:
    pred, num_tiles = overlap_tile_predict(model, img, tile_size=tile_size, pad_size=PAD_SIZE, batch_size=batch_size)
  else:
    pred = _predict(model, np.asarray([img]))
    pred = pred[0]
    tile_size = resize_shape
    num_tiles = 1
  
  if pred.shape[:2] != ori_shape:
    pred = tf.image.resize(pred, (ori_shape[0], ori_shape[1])).numpy()

  if DEBUG:
    shape_info = (ori_shape, resize_shape, resize_shape[0] / ori_shape[0], tile_size, resize_shape[0] / tile_size[0], num_tiles)
    print(shape_info)
    #shapes.add(shape_info)

# #   print(shapes)
#   global max_patches, num_shapes
  
#   if num_tiles > max_patches:
#     max_patches = num_tiles
# #     print('max_patches:', max_patches)
# #     print(ori_shape, resize_shape, num_tiles)
# #     print(shapes)
#   else:
#     if len(shapes) > num_shapes:
#       print(shapes)
    
#   num_shapes = len(shapes)
    
  pred = pred.argmax(axis=-1)
  f = np.vectorize(lambda x: m[x])
  pred = f(pred).astype(np.uint8)

  # # just for safe will not go here
  # if pred.shape[:2] != ori_shape:
  #   # print(pred.shape, ori_shape)
  #   pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)

#   print('pred_shape', pred.shape[:2])

  assert pred.shape[:2] == ori_shape, f'{pred.shape} {ori_shape}'
  index, _ = os.path.splitext(os.path.basename(input_path))
  cv2.imwrite(os.path.join(output_dir, f'{index}.png'), pred)