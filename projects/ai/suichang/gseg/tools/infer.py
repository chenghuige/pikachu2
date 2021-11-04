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
from gseg.util import inference

def main(_):
  FLAGS.write_test_results = True
  FLAGS.write_inter_results = False
  config.init()
  # 必须放到最前
  mt.init()

  global strategy, model
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    if FLAGS.data_version == 2:
      files = gezi.list_files('../input/quarter/tfrecords/train/*/*')
    else:
      files = gezi.list_files('../input/tfrecords/train/*/*')

    if FLAGS.parts:
      start, end = gezi.get_fold(len(test_files), FLAGS.parts, FLAGS.part)
      files = files[start:end]

    assert files, files

    dataset = Dataset('test').make_batch(FLAGS.batch_size, files)
    examples = mt.get_num_records(files)
    steps = -(-examples // FLAGS.batch_size)

    model = mt.load_model(FLAGS.pretrain)
    print(model.summary())
    inference(dataset, model, steps, examples, FLAGS.model_dir, desc='inference')


if __name__ == '__main__':
  app.run(main)  
