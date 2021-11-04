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
import tensorflow_addons as tfa

import gezi
logging = gezi.logging

from projects.feed.rank.src import config
from projects.feed.rank.src import util
from model import *
import model as base
import loss
import evaluate as ev
import husky

from tqdm import tqdm

STEPS_PER_EPOCH = 1000

def main(_):
  FLAGS.input = '/search/odin/publicData/CloudS/yuwenmengke/rank_0804_so/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020062402'
  FLAGS.valid_input = FLAGS.input
  #FLAGS.valid_input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051522'

  #FLAGS.input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520'
  #inputs = [
  #          '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520',
  #          '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051521',
  #          '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051522',
  #         ]
  #inputs = [x.replace('/home/gezi/data/rank', '/search/odin/publicData/CloudS/libowei/rank4') for x in inputs]
  #FLAGS.input = ','.join(inputs)
  #FLAGS.valid_input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051523'

  FLAGS.model_dir = '/tmp/melt'
  FLAGS.model = 'Model'
  FLAGS.hash_embedding_type = 'QREmbedding'
  FLAGS.batch_size = 512

  FLAGS.feature_dict_size=20000000 
  FLAGS.num_feature_buckets=3000000
  FLAGS.fields_pooling='dot'
  # FLAGS.wide_only = True
  FLAGS.keras = True
  FLAGS.shuffle = False

  FLAGS.optimizer = 'lazyadam'

  config.init()
  fit = melt.fit

  loss.init()
  melt.init()

  Dataset = util.prepare_dataset()

  model_name = FLAGS.model

  eval_fn, eval_keys = util.get_eval_fn_and_keys()
  valid_write_fn = ev.valid_write
  out_hook = ev.out_hook

  dataset = Dataset('train').make_batch(FLAGS.batch_size)
  eval_dataset = Dataset('valid').make_batch(FLAGS.batch_size, repeat=False)
  valid_dataset = Dataset('valid').make_batch(FLAGS.batch_size, repeat=True)
  strategy = melt.distributed.get_strategy()
  # if melt.distributed.has_strategy():
  #   dataset = strategy.experimental_distribute_dataset(dataset)
  # train_data_iter = iter(dataset) 

  # print(next(iter(dataset))
  K.set_learning_phase(1)

  with strategy.scope():
    model = getattr(base, FLAGS.model)() 
    melt.init_model(model, Dataset)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # def calc_loss(y_true, y):
    #   # tf.print(tf.shape(y_true), tf.shape(y))
    #   loss = tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y)
    #   # tf.print(loss)
    #   return loss
    # loss_fn = calc_loss
    optimizer=tfa.optimizers.LazyAdam(lr=1e-3, epsilon=1e-6)
    model.compile(
      loss=loss_fn,
      optimizer=optimizer,
      metrics=['acc']
     )

    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    train_auc = tf.keras.metrics.AUC()
    train_loss = tf.keras.metrics.Sum()

  # # # TODO 为什么加了 tf.function 反而更慢 因为 model内部 list  +[] 等python操作吗 
  # # # 加了2it/s 不加 7it/s 都远远低于目前tf1 graph模式的28it/s
  # # 不过目前tf2.3版本不加 tf.function 速度以及和 tf1一致了 但是不知道 eager/train.py 的loop方式为何慢
  # # @tf.function
  # def train_step(data_iter):
  #     def train_step_fn(item):
  #         x, y_true = item 
  #         with tf.GradientTape() as tape:
  #             y = model(x)
  #             loss = loss_fn(y_true, y)
  #             # print(loss)
  #             # print(y_true, y)
  #         grads = tape.gradient(loss, model.trainable_variables)
  #         grads_and_vars = zip(grads, model.trainable_variables)
  #         grads_and_vars = [(tf.convert_to_tensor(g), v) for g, v in grads_and_vars]
  #         optimizer.apply_gradients(grads_and_vars)
  #         # train_accuracy.update_state(y_true, y)
  #         # train_auc.update_state(y_true, y)
  #         # train_loss.update_state(loss)
  #     # for _ in tf.range(STEPS_PER_EPOCH):

  #     for i in tqdm(range(STEPS_PER_EPOCH), ascii=True):
  #         if i == 1:
  #           melt.print_model(model)
  #         if melt.distributed.has_strategy():
  #           strategy.experimental_run_v2(train_step_fn, next(data_iter))
  #         else:
  #           train_step_fn(next(data_iter))

  # # with gezi.Timer('train', print_fn=print):
  # #   train_step(train_data_iter)

  # FLAGS.num_train = len(Dataset('train'))
  # FLAGS.num_valid = len(Dataset('valid'))

  eval_fn, eval_keys = util.get_eval_fn_and_keys()
  valid_write_fn = ev.valid_write
  out_hook = ev.out_hook

  # callbacks = [keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir, profile_batch=(5,20))]
  callbacks = []
    
  # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss_fn = model.get_loss()

  # TODO FIXME tf2 not work in graph mode now for melt.fit
  with gezi.Timer('train', print_fn=print):
    # model.fit(dataset, epochs=1, steps_per_epoch=STEPS_PER_EPOCH)
    fit(model, loss_fn, 
        Dataset,
        #dataset=dataset, 
        #eval_dataset=eval_dataset, 
        #valid_dataset=valid_dataset,
        eval_fn=eval_fn,
        eval_keys=eval_keys,
        valid_write_fn=valid_write_fn,
        out_hook=out_hook,
        callbacks=callbacks,
       )
  

if __name__ == '__main__':
  app.run(main)   
