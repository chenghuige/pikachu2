#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige  
#          \date   2020-10-27 02:13:23.131541
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob
import cv2
from PIL import Image
import numpy as np

from gezi import tqdm
from gezi.metrics.image.semantic_seg import Evaluator
import gezi
 
pred_dir = sys.argv[1]

SRC = 'v2'

if len(sys.argv) > 2:
  SRC = sys.argv[2]
print('SRC:', SRC)

ROUND = 2
if len(sys.argv) > 3:
  ROUND = int(sys.argv[3])

if SRC == 'v2':
  #17 but 5,6 missing
  CLASSES = ['water', 'track_road', 'build', 'track_airport', 'other_park', 'other_playground', 'arable_natural', 'arable_greenhouse',
             'grass_natural', 'grass_greenbelt', 'forest_natural', 'forest_planted', 'bare_natural', 'bare_planted', 'other_other']
  label_dir = '../input/quarter/train/label'
elif SRC == 'v1':
  CLASSES = ['water', 'track', 'build', 'arable', 'grass', 'forest', 'bare', 'other']
  label_dir = '../input/train/label'
  if ROUND == 2:
    m2 = {
      0: 0,
      1: 1,
      2: 2,
      3: 1,
      4: 7,
      5: 7,
      6: 3,
      7: 3,
      8: 4,
      9: 4,
      10: 5,
      11: 5,
      12: 6,
      13: 6,
      14: 7,
    }
  else:
    m2 = dict(zip(range(8), range(8)))
elif SRC == 'ccf': # ccf data
  # 植被（标记1）、道路（标记2）、建筑（标记3）、水体（标记4）以及其他(标记0)
  CLASSES = ['background', 'vegetation', 'road', 'build', 'water']
  label_dir = '../input/ccf_remote_dataset/train/label'
  if ROUND == 2:
    m2 = {
      0: 4,
      1: 2,
      2: 3,
      3: 2,
      4: 0,
      5: 0,
      6: 1,
      7: 1,
      8: 1,
      9: 1,
      10: 1,
      11: 1,
      12: 1,
      13: 1,
      14: 0,
    }
  else:
    m2 = {
      0: 4,
      1: 2,
      2: 3,
      3: 1,
      4: 0,
      5: 0,
      6: 0,
      7: 0,
    }
elif SRC == 'baidu': # baidu
  CLASSES = ['build', 'arable', 'forest', 'water', 'track', 'grass', 'other']
  label_dir = '../input/baidu/train_data/lab_train'
  if ROUND == 2:
    m2 = {
      0: 3,
      1: 4,
      2: 0,
      3: 4,
      4: 6,
      5: 6,
      6: 1,
      7: 1,
      8: 5,
      9: 5,
      10: 2,
      11: 2,
      12: 6,
      13: 6,
      14: 6,
    }
  else:
    m2 = {
      0: 3,
      1: 4,
      2: 0,
      3: 1,
      4: 5,
      5: 2,
      6: 6,
      7: 6,
    }
else:
  raise ValueError(SRC)

NUM_CLASSES = len(CLASSES)

if len(sys.argv) > 4:
  label_dir = sys.argv[4]

print('label_dir:', label_dir)

m = {}
for i in range(17):
  if i < 4:
    m[i + 1] = i
  else:
    m[i + 1] = i - 2

key_metric = 'FWIoU'

pred_files = glob.glob(f'{pred_dir}/*')

evaluator = Evaluator(CLASSES)
class_mious = 0.

t = tqdm(enumerate(pred_files), total=len(pred_files), ascii=True, desc= 'eval')
for i, pred_file in t:
  file_name = os.path.basename(pred_file)
  label_file = f'{label_dir}/{file_name}'
  try:
    if SRC != 'baidu':
      label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED).astype(np.int32)
    else:
      label = np.asarray(Image.open(label_file))
    pred = cv2.imread(pred_file,cv2.IMREAD_UNCHANGED).astype(np.int32)    
  except Exception:
    print(label_file)
    continue

  if SRC == 'v2':
    f = np.vectorize(lambda x: m[x])
    label = f(label).astype(np.uint8)
    pred = f(pred).astype(np.uint8)
  else:
    if SRC == 'v1':
      label = (label / 100 - 1).astype(np.uint8)
      # pred = (pred / 100 - 1).astype(np.uint8)
    elif SRC == 'baidu':
      f = np.vectorize(lambda x: 0 if x == 255 else x)
      label = f(label).astype(np.uint8)

    f = np.vectorize(lambda x: m[x])
    pred = f(pred).astype(np.uint8)
    f = np.vectorize(lambda x: m2[x])
    pred = f(pred).astype(np.uint8)

  evaluator.eval_each(label[np.newaxis,:, :], pred[np.newaxis,:, :], metric=key_metric)

  binary_label = np.bincount(label.reshape(-1), minlength=NUM_CLASSES).astype(np.bool)
  binary_pred = np.bincount(pred.reshape(-1), minlength=NUM_CLASSES).astype(np.bool)
  intersections = binary_label * binary_pred
  intersection = np.sum(intersections, axis=-1)
  union = np.sum(binary_label, axis=-1) + np.sum(binary_pred, axis=-1) - intersection
  class_miou = np.mean(intersection / union)

  class_mious += class_miou

  t.set_postfix({key_metric: evaluator.eval_once(key_metric), 'IMAGE/CLASS/MIoU': class_mious / (i + 1)})

res = evaluator.eval_once()
res['IMAGE/CLASS/MIoU'] = class_mious / i
print(gezi.FormatDict(res))