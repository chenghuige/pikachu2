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

from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import backend as K

import gezi
logging = gezi.logging

from projects.feed.rank.src import config
from projects.feed.rank.src import util
from model import *
import model as base
import loss
import evaluate as ev

def main(_):
  # 一些FLAGS配置
  argv = open('./flags/tf/dlrm/command.txt').readline().strip().replace('data_version=2', 'data_version=1').split()
  FLAGS(argv)
  FLAGS.loop_train = False

  FLAGS.model_dir = '/tmp/melt'
  os.system('rm -rf %s' % FLAGS.model_dir)

  # 会读取这个3个路径下面的 总共 3 * 50 = 150个 tfrecord文件
  inputs = [
          '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520',
          '/search/odin/publicData/CloudS/libowei/rank4/newmse/data/video_hour_newmse_v1/tfrecords/2020051520',
          '/search/odin/publicData/CloudS/libowei/rank4/shida/data/video_hour_shida_v1/tfrecords/2020051520',
         ]

  FLAGS.input = ','.join(inputs)

  # 训练用到的所有150个tfrecord文件
  files = gezi.list_files(FLAGS.input)

  # valid 暂时不用,只做训练
  FLAGS.valid_input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051522'
  #FLAGS.valid_input = '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051523'

  # FLAGS.model = 'Model'
  # FLAGS.hash_embedding_type = 'QREmbedding'
  FLAGS.batch_size = max(FLAGS.batch_size, 512)
  # FLAGS.use_all_data = True
  # FLAGS.train_only = True

  # FLAGS.feature_dict_size= 20000000
  # FLAGS.num_feature_buckets=3000000
  # FLAGS.fields_pooling='dot'
  # FLAGS.use_weight = False
  # FLAGS.optimizer = 'lazyadam'
  #FLAGS.optimizer = 'sgd'
  # FLAGS.write_valid = True
  
  config.init()
  fit = melt.fit

  loss.init()

  # init 主要做一些初始化 比如选择最空闲的gpu logging初始化 模型路径 等等
  melt.init()

  # 定义了tfrecord读取的parse格式 可以看 tfrecord_dataset.py
  Dataset = util.prepare_dataset()

  model_name = FLAGS.model
  model = getattr(base, model_name)() 
  loss_fn = loss.get_loss_fn(model)
  optimizer = tf.contrib.opt.LazyAdamOptimizer()

  # sess 已经在melt.init构建好 如果不需要 melt.init() 也可以去掉melt.init() 这里使用 sess = tf.Session()
  sess = melt.get_session()
  dataset = Dataset('train')
  total = len(dataset)
  it = dataset.make_batch(batch_size=FLAGS.batch_size, filenames=files, repeat=False)
  x, y = it.get_next()
  num_steps = -int(-total // FLAGS.batch_size)

  # 表示train状态   
  K.set_learning_phase(1)
  y_ = model(x)
  train_loss = loss_fn(y, y_, x, model)
  train_op = optimizer.minimize(train_loss)
  
  init_op = tf.group(tf.compat.v1.global_variables_initializer(), 
                     tf.compat.v1.local_variables_initializer())
  sess.run(init_op)

  t = tqdm(range(num_steps), total=num_steps, desc='train', ascii=True)
  for i in t:
    train_loss_, _ = sess.run([train_loss, train_op])
    t.set_postfix({'loss': train_loss_})

if __name__ == '__main__':
  app.run(main)  
