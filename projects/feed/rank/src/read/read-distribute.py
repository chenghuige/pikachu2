#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from dataset import *

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt
import config

# 非eager模式 melt是按照step控制轮次的 使用的是 repeat 模式
# eager模式 melt 非repeate

# notcie input data 10 lines

def main(_):
  FLAGS.data_dir = '/home/gezi/new/temp/feed/rank/zjx_data_2'
  FLAGS.valid_input = os.path.join(FLAGS.data_dir, 'tfrecord/valid.small')
  FLAGS.train_input = os.path.join(FLAGS.data_dir, 'tfrecord/valid.small')
  print('-----', FLAGS.valid_input)
  FLAGS.batch_size = 2
  FLAGS.num_valid = 8
  config.init()
  melt.init()

  Dataset = TFRecordDataset if 'tfrecord' in FLAGS.train_input else TextDataset

  # tf.compat.v1.enable_v2_behavior()
  strategy = melt.distribution.get_strategy()
  with strategy.scope():
    # dataset = Dataset('valid')
    dataset = Dataset('train')

    iter = dataset.make_batch(repeat=True)
    # op = iter.get_next()

    print('---batch_size', dataset.batch_size, FLAGS.batch_size)  

    # sess = melt.get_session()

    num_steps = -int(-FLAGS.num_valid / melt.batch_size())
    print('----num_steps', num_steps) 
    # for epoch in range(1):
    #   for i in range(num_steps):
    #     batch = sess.run(op)
    #     data = [x.decode().split('\t')[0] for x in batch[0]['id']]
    #     print(epoch, i, data)
    
    for i, batch in enumerate(iter):
      print(batch)
      data = [x.decode().split('\t')[0] for x in batch[0]['id']]
      print(i, data)
      

if __name__ == '__main__':
  tf.compat.v1.app.run()  
