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

import copy
import numpy as np
import cv2
import tensorflow as tf
from tqdm.auto import tqdm

IMAGE_SIZE = (256, 256)
PAD_SIZE = (32, 32)

PAD = 0

BATCH_SIZE = 4

NUM_CLASSES = 15

m = {}
for i in range(NUM_CLASSES):
  if i < 4:
    m[i] = i + 1
  else:
    m[i] = i + 3

def _predict(model, img):
  pred = model.predict(img)
  pred = pred.argmax(axis=-1)
  pred = pred.astype(np.uint8)
  return pred

# from https://github.com/PaddlePaddle/PaddleX/blob/4a6c125339d7c05550d470a413a947967500fd44/paddlex/cv/models/deeplabv3p.py
def overlap_tile_predict(model, image, tile_size=IMAGE_SIZE, pad_size=PAD_SIZE, batch_size=BATCH_SIZE):
  """有重叠的大图切小图预测。
  Args:
      img_file(str|np.ndarray): 预测图像路径，或者是解码后的排列格式为（H, W, C）且类型为float32且为RGB格式的数组。
      tile_size(list|tuple): 滑动窗口的大小，该区域内用于拼接预测结果，格式为（W, H）。默认值为[512, 512]。
      pad_size(list|tuple): 滑动窗口向四周扩展的大小，扩展区域内不用于拼接预测结果，格式为（W, H）。默认值为[64，64]。
      batch_size(int)：对窗口进行批量预测时的批量大小。默认值为32
  Returns:
      dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
          像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
  """

  height, width, channel = image.shape
  image_tile_list = list()

  num_classes = model.output.shape[-1]

  print('imaage shape for tile predict', image.shape, 'num_output_classes', num_classes)
  # Padding along the left and right sides
  if pad_size[0] > 0:
    left_pad = cv2.flip(image[0:height, 0:pad_size[0], :], 1)
    right_pad = cv2.flip(image[0:height, -pad_size[0]:width, :], 1)
    padding_image = cv2.hconcat([left_pad, image])
    padding_image = cv2.hconcat([padding_image, right_pad])
  else:
    import copy
    padding_image = copy.deepcopy(image)

  # Padding along the upper and lower sides
  padding_height, padding_width, _ = padding_image.shape
  if pad_size[1] > 0:
    upper_pad = cv2.flip(
        padding_image[0:pad_size[1], 0:padding_width, :], 0)
    lower_pad = cv2.flip(
        padding_image[-pad_size[1]:padding_height, 0:padding_width, :],
        0)
    padding_image = cv2.vconcat([upper_pad, padding_image])
    padding_image = cv2.vconcat([padding_image, lower_pad])

  # crop the padding image into tile pieces
  padding_height, padding_width, _ = padding_image.shape

  print(padding_image.shape)

  hws = []
  # print('------abc', height // tile_size[1] + 1,  width // tile_size[0] + 1)
  assert height % tile_size[1] == 0
  assert width % tile_size[0] == 0

  print('hscale', height / tile_size[1], 'wscale', width / tile_size[0])

  for h_id in range(0, int(height / tile_size[1])):
    for w_id in range(0, int(width / tile_size[0])):
      hws.append([h_id, w_id])

  for h_id, w_id in tqdm(hws, desc='sub_image', ascii=True, leave=False):
    left = w_id * tile_size[0] 
    upper = h_id * tile_size[1] 
    right = min(left + tile_size[0] + pad_size[0] * 2,
                padding_width)
    lower = min(upper + tile_size[1] + pad_size[1] * 2,
                padding_height)
    image_tile = padding_image[upper:lower, left:right, :]
    # print(left, upper, right, lower, image_tile.shape)
    image_tile_list.append(image_tile)

  # predict
  score_map = np.zeros((height, width, num_classes), dtype=np.float32)

  num_tiles = len(image_tile_list)
  for i in tqdm(range(0, num_tiles, batch_size), desc='predict', ascii=True, leave=False):
    begin = i
    end = min(i + batch_size, num_tiles)
    # print([x.shape for x in image_tile_list[begin:end]])
    res = model.predict(np.asarray(image_tile_list[begin:end]))
    for j in range(begin, end):
      h_id = j // int(width / tile_size[1])
      w_id = j % int(width / tile_size[0])
      left = w_id * tile_size[0]
      upper = h_id * tile_size[1]
      right = min((w_id + 1) * tile_size[0], width)
      lower = min((h_id + 1) * tile_size[1], height)
      tile_score_map = res[j - begin]
      tile_upper = pad_size[1]
      tile_lower = tile_score_map.shape[0] - pad_size[1]
      tile_left = pad_size[0]
      tile_right = tile_score_map.shape[1] - pad_size[0]
      score_map[upper:lower, left:right, :] = \
          tile_score_map[tile_upper:tile_lower, tile_left:tile_right, :]

  pred = score_map.argmax(axis=-1)
  f = np.vectorize(lambda x: m[x])
  pred = f(pred).astype(np.uint8)
  return pred

def predict(model, input_path, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  ori_shape = img.shape[:2]
  print('ori shape', ori_shape)

  resize_shape = None
  if img.shape[:2] != IMAGE_SIZE:
    resize_h = round(ori_shape[0] / IMAGE_SIZE[0] + 0.001) * IMAGE_SIZE[0]
    resize_w = round(ori_shape[1] / IMAGE_SIZE[1] + 0.001) * IMAGE_SIZE[1]
    resize_shape = (resize_h, resize_w)

    print('resize shape', resize_shape)
    if resize_shape != ori_shape:
      img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)

  pred = overlap_tile_predict(model, img)

  print('pred_shape', pred.shape[:2])

  if resize_shape and resize_shape != ori_shape:
    pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)

  assert pred.shape[:2] == ori_shape, f'{pred.shape} {ori_shape}'
  index, _ = os.path.splitext(os.path.basename(input_path))
  cv2.imwrite(os.path.join(output_dir, f'{index}.png'), pred)