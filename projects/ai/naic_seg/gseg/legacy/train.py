#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   simple-train.py
#        \author   chenghuige  
#          \date   2020-09-28 16:10:12.412785
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

import gezi
import melt as mt
from config import *
from dataset import Dataset
from evaluate import eval
from projects.ai.naic2020_seg.baseline.model import unet

class Model(mt.Model):
  def __init__(self, model, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.model = model

  def __call__(self, x, training=False):
    return self.model(x, training=training)

def main(_):
  FLAGS.batch_parse = False
  FLAGS.static_input = True

  FLAGS.train_input = '../input/tfrecords/train/*/*'
  if FLAGS.fold is None:
    FLAGS.fold = 0

  mt.init()

  # if not FLAGS.fold:
  #   FLAGS.fold = 0

  # files = gezi.list_files(f'../input/tfrecords/train/*/*')
  # valid_files = gezi.list_files(f'../input/tfrecords/train/{FLAGS.fold}/*')
  # train_files = [x for x in files if x not in valid_files]
  # dtrain = Dataset('train').make_batch(FLAGS.batch_size, train_files)
  # deval = Dataset('valid').make_batch(FLAGS.eval_batch_size, valid_files, repeat=False)
  # dvalid = Dataset('valid').make_batch(FLAGS.eval_batch_size, valid_files, repeat=True, cache=True)

  # FLAGS.num_train = mt.get_num_records(train_files)
  # FLAGS.num_valid = mt.get_num_records(valid_files)

  strategy = mt.distributed.get_strategy()

  # with strategy.scope():
  #   model = unet(8)
  #   mt.fit(model, 
  #         tf.keras.losses.SparseCategoricalCrossentropy(),
  #         Dataset=None,
  #         dataset=dtrain,
  #         eval_dataset=deval,
  #         valid_dataset=dvalid,
  #         eval_fn=eval
  #         )

  #model.compile(loss='sparse_categorical_crossentropy',
  #              optimizer='sgd',
  #              metrics=['accuracy'])
  #model.fit(dtrain, steps_per_epoch=200, epochs=4)

  with strategy.scope():
    model = unet(NUM_CLASSES)
    model = Model(model)
    mt.fit(model, 
          tf.keras.losses.SparseCategoricalCrossentropy(),
          Dataset=Dataset,
          eval_fn=eval
          )

if __name__ == '__main__':
  app.run(main)  
