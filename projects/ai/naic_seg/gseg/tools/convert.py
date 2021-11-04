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
os.environ['SM_FRAMEWORK'] = 'tf.keras'
sys.path.append('..')

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
from gezi import logging
import melt as mt
from gseg import config
from gseg.config import *
from gseg.dataset import Dataset
from gseg.evaluate import get_eval_fn
from gseg.util import get_infer_fn
from gseg.model import get_model
from gseg.loss import get_loss_fn
from gseg.metrics import get_metrics

strategy = None
model = None

def fit(dry_run=False):
  with strategy.scope():
    mt.fit(model, 
          #  loss_fn=get_loss_fn(),
           loss_fn=model.get_loss(),
           Dataset=Dataset,
           metrics=get_metrics(),
           eval_fn=get_eval_fn(),
           inference_fn=get_infer_fn(),
           dry_run=dry_run
          )

def main(_):
  FLAGS.load_by_name = False
  if FLAGS.pretrain:
    FLAGS.mn = FLAGS.pretrain if not '/' in FLAGS.pretrain else os.path.basename(FLAGS.pretrain)
  config.init()
  # 必须放到最前
  mt.init()

  lock_file = f'{FLAGS.model_dir}/lock.txt'
  if os.path.exists(lock_file):
  # if os.path.exists(f'{FLAGS.model_dir}/model.h5'):
    logging.warning(f'Already exists {FLAGS.model_dir}/model.h5, remove {FLAGS.model_dir} first or add --clear_first')
    exit(0)
  gezi.write_to_txt('lock', lock_file)

  NUM_VALID = FLAGS.num_valid

  global strategy, model
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = get_model(FLAGS.model)

  logging.info(f'save {FLAGS.model_dir}/model.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = False
  FLAGS.num_valid = NUM_VALID
  fit()
  metrics = gezi.get('metrics')
  #assert metrics['Metric/FWIoU'] > 0.75, FLAGS.pretrain
  mt.save_model(model, f'{FLAGS.model_dir}/model.h5')

  if gezi.get_env('FAST'):
    NUM_VALID = 1000

  logging.info(f'save {FLAGS.model_dir}/model_lr.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = True
  FLAGS.tta_fns = ['flip_left_right']
  FLAGS.tta_use_original = False
  FLAGS.num_valid = NUM_VALID
  fit()
  mt.save_model(model, f'{FLAGS.model_dir}/model_lr.h5')

  logging.info(f'save {FLAGS.model_dir}/model_ud.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = True
  FLAGS.tta_fns = ['flip_up_down']
  FLAGS.tta_use_original = False
  FLAGS.num_valid = NUM_VALID
  fit()
  mt.save_model(model, f'{FLAGS.model_dir}/model_ud.h5')

  logging.info(f'save {FLAGS.model_dir}/model_rot1.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = True
  FLAGS.tta_fns = ['rot90_1']
  FLAGS.tta_use_original = False
  FLAGS.num_valid = NUM_VALID
  fit()
  mt.save_model(model, f'{FLAGS.model_dir}/model_rot1.h5')

  logging.info(f'save {FLAGS.model_dir}/model_rot1.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = True
  FLAGS.tta_fns = ['rot90_1']
  FLAGS.tta_use_original = False
  FLAGS.num_valid = NUM_VALID
  fit()
  mt.save_model(model, f'{FLAGS.model_dir}/model_rot2.h5')

  logging.info(f'save {FLAGS.model_dir}/model_rot2.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = True
  FLAGS.tta_fns = ['rot90_2']
  FLAGS.tta_use_original = False
  FLAGS.num_valid = NUM_VALID
  fit()
  mt.save_model(model, f'{FLAGS.model_dir}/model_rot2.h5')

  logging.info(f'save {FLAGS.model_dir}/model_rot3.h5')
  FLAGS.custom_eval = False
  FLAGS.tta = True
  FLAGS.tta_fns = ['rot90_3']
  FLAGS.tta_use_original = False
  FLAGS.num_valid = NUM_VALID
  fit()
  mt.save_model(model, f'{FLAGS.model_dir}/model_rot3.h5')


if __name__ == '__main__':
  app.run(main)  
