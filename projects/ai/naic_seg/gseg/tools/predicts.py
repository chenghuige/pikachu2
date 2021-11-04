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
import tensorflow as tf

if os.environ.get('DEBUG') == '1':
  callbacks = None
  log_dir = '/tmp/melt'
  if os.path.exists(log_dir):
    os.system('rm -rf %s' % log_dir)
  os.makedirs(log_dir, exist_ok=True)
  print('DEBUG:1')
else:
  print('DEBUG:0')

def generate_outputs(pyfile_path, input_paths, output_dir):
  pver = os.environ.get('PVER')
  if not pver or pver == '0': 
    predict_py = pyfile_path + '.model_predict'
    print('PVER:default')
  else:
    predict_py = pyfile_path + f'.v{pver}.model_predict'
    print(f'PVER:{pver}')
    
  define_py = pyfile_path + '.model_define'
  init_model = getattr(importlib.import_module(define_py), 'init_model')
  predict = getattr(importlib.import_module(predict_py), 'predict')

  np.random.shuffle(input_paths)

  model = init_model()

  if os.environ.get('DEBUG') == '1':
    tf.profiler.experimental.start(log_dir)
  done = False
  for i, input_path in tqdm(enumerate(input_paths), total=len(input_paths), ascii=True, desc='infer'):
    if os.environ.get('DEBUG') == '1' and max_profile and i == max_profile:
      done = True
      tf.profiler.experimental.stop()

    predict(model, input_path, output_dir)
  
  if os.environ.get('DEBUG') == '1' and not done:
    tf.profiler.experimental.stop()

def get_img_label_paths(images_path, labels_path):
  res = []
  for dir_entry in os.listdir(images_path):
    if os.path.isfile(os.path.join(images_path, dir_entry)):
      file_name, suffix = os.path.splitext(dir_entry)
      res.append((os.path.join(images_path, file_name + f"{suffix}"),
                  os.path.join(labels_path, file_name + ".png")))
  return res

def predicts(input_dir, output_dir, max_files=None, fold=1, folds=10, records=30):
  print(input_dir)
  print(output_dir)
  image_dir = input_dir
  label_dir = input_dir.replace('image', 'label')
  if os.path.exists(label_dir):
    imgs = get_img_label_paths(image_dir, label_dir)
    np.random.seed(12345)
    np.random.shuffle(imgs)

    input_paths = []
    for i in range(len(imgs)):
      # print(imgs[i][0], 'ccf' in imgs[i][0], os.path.basename(imgs[i][0]), fold + 1, os.path.basename(imgs[i][0]).startswith(f'{fold + 1}'))
      if 'ccf' in imgs[i][0] and os.path.basename(imgs[i][0]).startswith(f'{fold + 1}') or \
        'ccf' not in imgs[i][0] and i % records % folds == fold: 
        input_paths.append(imgs[i][0])
  else:
    input_paths = glob.glob(f'{input_dir}/*')

  if max_files:
    input_paths = input_paths[:max_files]
  print(input_paths[:5])

  pyfile_path = os.path.dirname(__file__)

  generate_outputs('infer', input_paths, output_dir)

max_files = 0
if len(sys.argv) > 3:
  max_files = int(sys.argv[3])

if os.environ.get('MAX_FILES'):
  max_files = int(os.environ.get('MAX_FILES'))
print('max_files:', max_files)

max_profile_ = os.environ.get('MAX_PROFILE')
max_profile =  int(max_profile_) if max_profile_ else max_files
print('max_profile:', max_profile)

fold = 1
if len(sys.argv) > 4:
  fold = int(sys.argv[4])

folds = 10
if len(sys.argv) > 5:
  folds = int(sys.argv[5])

records = 30
if len(sys.argv) > 6:
  records = int(sys.argv[6])

predicts(sys.argv[1], sys.argv[2], max_files=max_files, fold=fold, folds=folds, records=records) 
