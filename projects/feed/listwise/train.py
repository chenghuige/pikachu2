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

import sys
import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt

from projects.feed.rank.src import config
from projects.feed.rank.src import util
from model import Model
from dataset import Dataset
from evaluate import evaluate

#  def loss(y, y_, x, model):

def main(_):
  fit = melt.fit
  FLAGS.train_input = f'/home/gezi/tmp/rerank/tfrecords/train/{sys.argv[1]}'
  FLAGS.valid_input = f'/home/gezi/tmp/rerank/tfrecords/valid/{sys.argv[2]}'
  FLAGS.global_epoch = 0 
  FLAGS.global_step = 0 
  FLAGS.learning_rate = 0.1
  FLAGS.min_learning_rate = 1e-06
  FLAGS.optimizer = 'bert-adam'
  FLAGS.optimizer = 'adam'

  # FLAGS.vie = 0.1
  # tf.enable_eager_execution()
  melt.init()
  model = Model()
  fit(model,  
      loss_fn=tf.compat.v1.losses.sigmoid_cross_entropy,
      Dataset=Dataset,
      # eval_fn=evaluate,
      ) 

if __name__ == '__main__':
  app.run(main)  
