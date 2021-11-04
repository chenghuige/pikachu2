#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2019-07-26 23:14:46.546584
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'DenseNet121', '')
flags.DEFINE_string('dir', '../input/aptos2019-blindness-detection', '')
flags.DEFINE_string('pretrain_path', '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5', '')
flags.DEFINE_bool('random_brightness', True, '')
flags.DEFINE_bool('random_contrast', True, '')
flags.DEFINE_string('loss_type', 'classification', 
                    'classification, linear_regression, sigmoid_regression, ordinal_classification')
flags.DEFINE_integer('num_freeze_epochs', 2, '')
flags.DEFINE_bool('use_aug', True, '')


NUM_CLASSES = 5  
NUM_FOLDS = 5
RANDOM_STATE = 2019 
IMAGE_SIZE = 300 
NUM_EPOCHS = 50  
NUM_CHANNELS = 3

HEIGHT = 300
WIDTH = 300
IMAGE_SIZE = 300
DEPTH = 3


IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, DEPTH)