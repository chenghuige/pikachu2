#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-26 23:00:24.215922
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys 
# import os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt 
logging = melt.logging
# import numpy as np

from config import *

class Dataset(melt.Dataset):
  def __init__(self, subset='valid'):
    super(Dataset, self).__init__(subset)
    self.use_aug = subset == 'train' and FLAGS.use_aug

  def parse(self, example):
    features_dict = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)}

    features = tf.io.parse_single_example(serialized=example, features=features_dict)

    y = features['label']

    del features['label']
    x = features

    image = x['image']
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    image = self.preprocess(image)
    x['image'] = image

    logging.info('x', x, 'y', y)

    return x, y

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""

    image = tf.image.resize(image, [HEIGHT, WIDTH], method=0)    
    if self.use_aug:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      #... yes should do something like below.. but you will see with dataset.map.. not ok as summary without scope and finally graph has no these summaries
      # refer to https://stackoverflow.com/questions/47345394/image-summaries-with-tensorflows-dataset-api  TODO FIXME
      # tf.summary.image('image', image)
      #print('--------', image)
      #image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
      image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
      if FLAGS.random_brightness:
        image = tf.image.random_brightness(image, max_delta=63)
      if FLAGS.random_contrast:
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
      # tf.summary.image('image/distort', image)

    image = tf.cast(
        tf.reshape(image, [HEIGHT, WIDTH, DEPTH]),
        tf.float32)

    return image
