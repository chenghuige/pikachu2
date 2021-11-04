#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train_simclr.py
#        \author   chenghuige  
#          \date   2020-11-14 16:11:51.356766
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
import melt as mt
from gseg import config
from gseg.config import *
from gseg.dataset import Dataset
from gseg.simclr.model import Model

def main(_):
  # FLAGS.train_input = '../input/tfrecords/train/*/*'
  GCS_ROOT = '../input'
  GCS_PATH = '../input/tfrecords'
  # FLAGS.input = f'{GCS_ROOT}/quarter/tfrecords/train/*/*,{GCS_PATH}/train/*/*,{GCS_PATH}/test_A/*/*,{GCS_PATH}/test_B/*/*'
  FLAGS.input = f'{GCS_ROOT}/quarter/tfrecords/train/*/*'
  FLAGS.valid_input = f'{GCS_PATH}/test_A/1/*'
  
  # FLAGS.fold = 1
  FLAGS.model_dir = '../working/simclr'
  FLAGS.print_depth = 1
  FLAGS.no_labels = True
  FLAGS.augment_level = 4
  FLAGS.sharpen_rate = 0.3
  FLAGS.blur_rate = 0.3
  
  config.init()
  
  FLAGS.aug_train_image = False

  # 必须放到最前
  mt.init()

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = Model(backbone=FLAGS.backbone, weights=FLAGS.backbone_weights, image_size=FLAGS.ori_image_size)
    loss_fn=model.get_loss() 
    mt.fit(model, 
           loss_fn=loss_fn,
           Dataset=Dataset,
          )

  mt.save_model(model.get_model(), os.path.join(FLAGS.model_dir, 'model.h5'))
  mt.save_model(model.model, os.path.join(FLAGS.model_dir, f'SimCLR_{FLAGS.backbone}.h5'))

if __name__ == '__main__':
  app.run(main)  
  
