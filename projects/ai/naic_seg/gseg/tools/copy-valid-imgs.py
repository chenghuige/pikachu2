#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predicts.py
#        \author   chenghuige  
#          \date   2020-10-27 00:14:06.539840
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import importlib
import glob
import numpy as np
from tqdm.auto import tqdm
import shutil
import gezi

def get_img_label_paths(images_path, labels_path):
  res = []
  for dir_entry in os.listdir(images_path):
    if os.path.isfile(os.path.join(images_path, dir_entry)):
      file_name, _ = os.path.splitext(dir_entry)
      res.append((os.path.join(images_path, file_name + ".tif"),
                  os.path.join(labels_path, file_name + ".png")))
  return res

def deal(input_dir, out_dir, fold=1, folds=10, records=30):
  gezi.try_mkdir(out_dir)
  image_dir = input_dir
  label_dir = input_dir.replace('image', 'label')
  print(image_dir)
  print(label_dir)
  assert os.path.exists(label_dir)
  imgs = get_img_label_paths(image_dir, label_dir)
  np.random.seed(12345)
  np.random.shuffle(imgs)

  input_paths = []
  for i in tqdm(range(len(imgs)), desc='loop-files'):
    if i % records % folds == fold: 
      input_paths.append(imgs[i][0])
 
  #print(input_paths)
  for input_path in tqdm(input_paths, desc='copy-files'):
    shutil.copy2(input_path, out_dir)

fold = 1
if len(sys.argv) > 3:
  fold = int(sys.argv[3])

deal(sys.argv[1], sys.argv[2], fold=fold, folds=10, records=30) 

