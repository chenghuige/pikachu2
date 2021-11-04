#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tf2-train.py
#        \author   chenghuige  
#          \date   2020-04-25 15:43:47.905524
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras
from tensorflow.keras import backend as K

import gezi
logging = gezi.logging

import melt
import config
from config import *
import model as base
import loss
from evaluate import evaluate
from dataset import Dataset
from tqdm import tqdm

def main(_):
  FLAGS.torch = True

  config.init()
  fit = melt.fit

  melt.init()

  model_name = FLAGS.model

  print(FLAGS.train_input)
  print(gezi.list_files(FLAGS.train_input))
  files = [x for x in gezi.list_files(FLAGS.train_input) if not os.path.basename(x).startswith(f'record_{FLAGS.fold}')]
  dataset = Dataset('train').make_batch(FLAGS.batch_size, files)
  # eval_dataset = Dataset('valid').make_batch(FLAGS.batch_size, repeat=False)
  # valid_dataset = Dataset('valid').make_batch(FLAGS.batch_size, repeat=True)
  num_examples = melt.get_num_records(files)
  print('num_examples', num_examples)
 
  steps_per_epoch = -(-num_examples // FLAGS.batch_size)
  print('steps_per_epoch', steps_per_epoch)

  strategy = melt.distributed.get_strategy()
  if melt.distributed.has_strategy():
    dataset = strategy.experimental_distribute_dataset(dataset)
  train_data_iter = iter(dataset) 

  K.set_learning_phase(1)

  with strategy.scope():
    model = getattr(base, FLAGS.model)() 
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer=tf.keras.optimizers.Adam(lr=3e-5, epsilon=1e-8)
    model.compile(
      loss=loss_fn,
      optimizer=optimizer,
      metrics=['acc']
     )

  # 但还是比model.fit 慢很多啊。。。
  @tf.function # OOM without tf.function
  def train_step(data_iter):
    def train_step_fn(item):
      x, y_true = item 
      with tf.GradientTape() as tape:
        y = model(x)
        loss = loss_fn(y_true, y)
        # gezi.set('loss', loss)
        tf.print('loss:', loss, end='')
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    t = tqdm(range(steps_per_epoch), ascii=True)
    for i in t:
      # t.set_postfix(loss=loss))
      if i == 1:
        melt.print_model(model)
      if melt.distributed.has_strategy():
        strategy.experimental_run_v2(train_step_fn, next(data_iter))
      else:
        train_step_fn(next(data_iter))

  with gezi.Timer('train', print_fn=print):
    train_step(train_data_iter)


if __name__ == '__main__':
  app.run(main)   
