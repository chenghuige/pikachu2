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

IMAGE_SIZE = (256, 256)
PAD = 0

bs = os.environ.get('BATCH_SIZE')

try:
  BATCH_SIZE = int(bs) if bs else 4
except Exception as e:
  print(e)
  BATCH_SIZE = 4

print('batch_size:', BATCH_SIZE)

m = {}
for i in range(15):
  if i < 4:
    m[i] = i + 1
  else:
    m[i] = i + 3

tta = False
if os.environ.get('TTA') == '1':
  tta = True
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
  pred = pred.argmax(axis=-1)
  pred = pred.astype(np.uint8)
  return pred

def predict(model, input_path, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  ori_shape = img.shape[:2]
  resize_shape = None
  final_shape = None

  if img.shape[:2] != IMAGE_SIZE:
    scale = max(round(ori_shape[0] / (IMAGE_SIZE[0] - 2 * PAD)), 1)
    resize_len = scale *  (IMAGE_SIZE[0] - 2 * PAD) 
    resize_shape = (resize_len, resize_len)

    if resize_shape != ori_shape:
      img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)

    if PAD:
      img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)

    final_shape = img.shape[:2]

    # print('ori_shape:', ori_shape, 'resize_shape:', resize_shape, 'final_shape', final_shape)
  
  preds = []
  # print('------------abc', resize_shape, IMAGE_SIZE)
  if not resize_shape or resize_shape == IMAGE_SIZE:
    img = np.asarray([img])
    pred = _predict(model, img)
    preds.append(pred)
  else:
    assert resize_shape[0] % (IMAGE_SIZE[0] - 2 * PAD) == 0
    imgs = []
    i = 0
    hws = []
    for h in range(PAD, resize_shape[0] + PAD, (IMAGE_SIZE[0] - 2 * PAD)):
      for w in range(PAD, resize_shape[1] + PAD, (IMAGE_SIZE[1] - 2 * PAD)):
        hws.append((h,w))
    for h, w in tqdm(hws, desc='sub_image', ascii=True, leave=False):
        left = w - PAD
        upper = h - PAD
        right = min(w + IMAGE_SIZE[1] - PAD, final_shape[1])
        lower = min(h + IMAGE_SIZE[0] - PAD, final_shape[0])
        tile_img = img[upper:lower, left:right, :]
        # print(i, h, w, left, upper, right, lower, tile_img.shape)
        imgs.append(tile_img)
        if len(imgs) == BATCH_SIZE:
          pred = _predict(model, np.asarray(imgs))
          # print(pred.shape)
          preds.append(pred)
          imgs = []
        i += 1
    if imgs:
      pred = _predict(model, np.asarray(imgs))
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
        l_.append(preds[i][PAD:IMAGE_SIZE[0] - PAD, PAD:IMAGE_SIZE[0] - PAD])
        i += 1
      l.append(cv2.hconcat(l_))
    pred = cv2.vconcat(l)
  else:
    pred = preds[0]

  f = np.vectorize(lambda x: m[x])
  pred = f(pred).astype(np.uint8)

  # print(resize_shape)
  # print(ori_shape)
  if resize_shape and resize_shape != ori_shape:
    # print(pred.shape, ori_shape)
    pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)

  assert pred.shape[:2] == ori_shape, f'{pred.shape} {ori_shape}'
  index, _ = os.path.splitext(os.path.basename(input_path))
  cv2.imwrite(os.path.join(output_dir, f'{index}.png'), pred)