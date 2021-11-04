#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2020-09-28 16:10:12.412785
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob
from gezi import tqdm

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('clear_first', False, '')
flags.DEFINE_bool('fast', True, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_string('teacher', '', '')
flags.DEFINE_bool('adjust_preds', False, '')

def get_model(model_name):
  model = 'sm.Unet'
  if 'FPN' in model_name:
    model = 'sm.FPN'
  elif 'Linknet' in model_name:
    model = 'sm.Linknet'
  elif 'bread.deeplab' in model_name.lower():
    model = 'bread.DeepLabV3Plus'
  else:
    model = model_name.split('.')[0]
  print('model:', model)
  return model

def get_backbone(model_name):
  backbone = 'EfficientNetB4'
  assert 'EfficientNet' in model_name
  l = model_name.split('.')
  for x in l:
    if 'EfficientNet' in x:
      backbone = x
      break
  print('backbone:', backbone)
  return backbone

def get_size(model_name):
  if 'size288' in model_name:
    return '288,288'
  elif 'size320' in model_name:
    return '320,320'
  elif 'size352' in model_name:
    return '352,352'
  elif 'size512' in model_name:
    return '512,512'
  else:
    return '256,256'

def get_mrate(model_name):
  mrate = 0
  l = model_name.split('.')
  for x in l:
    if 'mrate' in x:
      mrate = int(x[len('mrate'):])
      break
  return mrate

# TODO bydefault should be largefilter=1, for unet
def get_largefilter(model_name):
  # largefilter = 0
  # l = model_name.split('.')
  # for x in l:
  #   if 'largefilter' in x:
  #     largefilter = 1
  #     break
  # return largefilter
  return 1

# for deeplab
def get_largeatrous(model_name):
  # last = model_name.split('.')[-1]
  # if not last.startswith('11') and not last.startswith('12'):
  #   return 0
  # elif last >= '1112':
  #   return 1
  # else:
  #   return 0
  return 1

def get_fpn_filters(model_name):
  # if model_name == 'sm.FPN.EfficientNetB7.augl3.200epoch':
  #   return 128
  # return 256
  return 256

def main(_):
  model_dir = sys.argv[1]
  if os.path.exists(f'{model_dir}/model.h5'):
    models = [model_dir]
  else:
    models = glob.glob(f'{model_dir}/*')
  model_names = [os.path.basename(m) for m in models]
  print(model_names)
  clear_first = FLAGS.clear_first
  fast = FLAGS.fast
  t = tqdm(zip(model_names, models), total=len(models), desc='convert')
  for mn, model in t:
    t.set_postfix({'model': mn})
    command = f'FAST={int(fast)} sh ./convert/common.sh --image_size={get_size(mn)} --model={get_model(mn)} --backbone={get_backbone(mn)} --mrate={get_mrate(mn)} --unet_large_filters={get_largefilter(mn)} --deeplab_large_atrous={get_largeatrous(mn)} --fpn_filters={get_fpn_filters(mn)} --pretrain={model} --mn={mn} --clear_first={int(clear_first)} --batch_size={FLAGS.batch_size} --teacher={FLAGS.teacher} --adjust_preds={int(FLAGS.adjust_preds)}'
    print(command)
    ret = os.system(command)
    if not ret == 0:
      exit(-1)

if __name__ == '__main__':
  app.run(main)  
