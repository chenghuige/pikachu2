#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
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

import gezi
logging = gezi.logging

from projects.feed.rank.src import config
from projects.feed.rank.src import util
from model import *
import model as base
import loss
import evaluate as ev

def main(_):
  FLAGS.input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520'
  FLAGS.valid_input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051522'
  FLAGS.model_dir = '/tmp/melt'
  FLAGS.model = 'Model'
  FLAGS.hash_embedding_type = 'QREmbedding'
  FLAGS.batch_size = 512

  FLAGS.feature_dict_size=20000000
  FLAGS.num_feature_buckets=3000000
  FLAGS.fields_pooling='dot'

  FLAGS.use_weight = False
  FLAGS.optimizer = 'lazyadam'

  FLAGS.write_valid = True
  
  config.init()
  fit = melt.fit

  loss.init()
  # melt.init()

  Dataset = util.prepare_dataset()

  device_setter = gezi.get('device_setter')
  with tf.Graph().as_default():
    with tf.device(device_setter):
      model_name = FLAGS.model
      with melt.distributed.get_strategy().scope():
        model = getattr(base, model_name)() 
      
    loss_fn = loss.get_loss_fn(model)

    dataset = Dataset('train')
    iter = dataset.make_batch(batch_size, inputs, repeat=True, initializable=True, 
                              hvd_shard=FLAGS.horovod_shard, simple_parse=FLAGS.simple_parse,
                              cache=FLAGS.cache or FLAGS.cache_train, cache_file=FLAGS.cache_file,
                              world_size=FLAGS.world_size, rank=FLAGS.local_rank)

  # eval_fn, eval_keys = util.get_eval_fn_and_keys()
  # valid_write_fn = ev.valid_write
  # out_hook = ev.out_hook

  # # for different vars with different optimizers, learning rates
  # variables_list_fn = util.get_variables_list

  # weights = None if not FLAGS.use_weight else 'weight'
  
  # # callbacks = [keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=1)]
  # callbacks = []

  # fit(model,  
  #     loss_fn,
  #     Dataset,
  #     variables_list_fn=variables_list_fn,
  #     eval_fn=eval_fn,
  #     eval_keys=eval_keys,
  #     valid_write_fn=valid_write_fn,
  #     weights=weights,
  #     out_hook=out_hook,
  #     callbacks=callbacks)   

if __name__ == '__main__':
  app.run(main)  
