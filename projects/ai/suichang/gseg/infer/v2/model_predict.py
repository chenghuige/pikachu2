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

# IMAGE_SIZE_ = os.environ.get('IMAGE_SIZE')
# IMAGE_SIZE_ = int(IMAGE_SIZE_) if IMAGE_SIZE_ else 256
# assert IMAGE_SIZE_ % 32 == 0
IMAGE_SIZE_ = 256
IMAGE_SIZE = (IMAGE_SIZE_, IMAGE_SIZE_)

# bs = os.environ.get('BATCH_SIZE')

# try:
#   BATCH_SIZE = int(bs) if bs else 4
# except Exception as e:
#   print(e)
#   BATCH_SIZE = 4

BATCH_SIZE = 4
print('batch_size:', BATCH_SIZE)

m = {}
for i in range(15):
  if i < 4:
    m[i] = i + 1
  else:
    m[i] = i + 3

# max_patches = 0

tta = False
# if os.environ.get('TTA') == '1':
#   tta = True
# print('tta:', tta)

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
  return pred

def get_size(height, width):
  h = round((height / 32) + 0.001) * 32
  w = round((width / 32) + 0.001) * 32
  return (h, w)

def predict(model, input_path, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  ori_shape = img.shape[:2]
  resize_shape = None
  final_shape = None

  # if ori_shape[0] * ori_shape[1] <= 512 * 512:
  #   resize_h, resize_w = get_size(ori_shape[0], ori_shape[1])
  #   resize_shape = (resize_h, resize_w)

  #   if resize_shape != ori_shape:
  #     img = tf.image.resize(img, (resize_shape[0], resize_shape[1])).numpy()
    
  #   pred = _predict(model, np.asarray([img]))
  #   pred = pred[0]
  # else:
    
  if img.shape[:2] != IMAGE_SIZE:
    h_scale = max(round(ori_shape[0] / IMAGE_SIZE[0] + 0.001), 1)
    w_scale = max(round(ori_shape[1] / IMAGE_SIZE[1] + 0.001), 1)
    # print('h_scale', h_scale, 'w_scale', w_scale)
    resize_h = h_scale * IMAGE_SIZE[0] 
    resize_w = w_scale * IMAGE_SIZE[1] 
    resize_shape = (resize_h, resize_w)
    # print('resize_shape', resize_shape)

    if resize_shape != ori_shape:
      # img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
      # img = cv2.resize(img, (resize_shape[1], resize_shape[0]))
      img = tf.image.resize(img, (resize_shape[0], resize_shape[1])).numpy()

    final_shape = img.shape[:2]

    # print('ori_shape:', ori_shape, 'resize_shape:', resize_shape, 'final_shape', final_shape)
  
  preds = []
  # print('------------abc', resize_shape, IMAGE_SIZE)
  if not resize_shape or resize_shape == IMAGE_SIZE:
    img = np.asarray([img])
    pred = _predict(model, img)
    # pred = model.predict(img)
    preds.append(pred)
  else:
    assert resize_shape[0] % IMAGE_SIZE[0] == 0
    imgs = []
    i = 0
    hws = []
    for h in range(0, resize_shape[0], IMAGE_SIZE[0]):
      for w in range(0, resize_shape[1], IMAGE_SIZE[1]):
        hws.append((h,w))
    
    # global max_patches
    # if len(hws) > max_patches:
    #   max_patches = len(hws)
    #   print('max_patches', max_patches)

    for h, w in tqdm(hws, desc='sub_image', ascii=True, leave=False):
        left = w 
        upper = h
        right = min(w + IMAGE_SIZE[1], final_shape[1])
        lower = min(h + IMAGE_SIZE[0], final_shape[0])
        tile_img = img[upper:lower, left:right, :]
        # print(i, h, w, left, upper, right, lower, tile_img.shape)
        imgs.append(tile_img)
        if len(imgs) == BATCH_SIZE:
          pred = _predict(model, np.asarray(imgs))
          # pred = model.predict(np.asarray(imgs))
          # print(pred.shape)
          preds.append(pred)
          imgs = []
        i += 1
    if imgs:
      # pred = model.predict(np.asarray(imgs))
      _predict(model, np.asarray(imgs))
      preds.append(pred)
      imgs = []

  # print('-----', len(preds), preds[0].shape)
  # print(preds)
  preds = np.concatenate(preds, axis=0)
  # print(preds.shape)
  # print(len(preds))
  # print('---------------')
  if len(preds) > 1:
    i = 0
    l = []
    for _ in range(0, resize_shape[0], IMAGE_SIZE[0]):
      l_ = []
      for _ in range(0, resize_shape[1], IMAGE_SIZE[1]):
        # l_.append(preds[i][:IMAGE_SIZE[0], :IMAGE_SIZE[0]])
        l_.append(preds[i])
        i += 1
      l.append(cv2.hconcat(l_))
    pred = cv2.vconcat(l)
  else:
    pred = preds[0]

  if resize_shape and resize_shape != ori_shape:
    pred = tf.image.resize(pred, (ori_shape[0], ori_shape[1])).numpy()

  pred = pred.argmax(axis=-1)
  pred = pred.astype(np.uint8)

  f = np.vectorize(lambda x: m[x])
  pred = f(pred).astype(np.uint8)

  # print(resize_shape)
  # print(ori_shape)
  # if resize_shape and resize_shape != ori_shape:
    # print(pred.shape, ori_shape)
    # pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)
    # pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]))
    # pred = tf.image.resize(pred[:, :, np.newaxis], (ori_shape[0], ori_shape[1])).numpy().squeeze(-1)

  assert pred.shape[:2] == ori_shape, f'{pred.shape} {ori_shape}'
  index, _ = os.path.splitext(os.path.basename(input_path))
  cv2.imwrite(os.path.join(output_dir, f'{index}.png'), pred)