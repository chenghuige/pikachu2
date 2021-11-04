#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2020-04-12 20:33:51.902319
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
from tensorflow.keras import backend as K
# import tensorflow_io as tfio
import numpy as np

import gezi
import melt as mt

def decode_image(image_data):
  image = tf.image.decode_image(image_data, channels=3)
  image = image[:, :, :3]
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [256, 256, 3])
  return image

def augment(image):
  if tf.random.uniform(()) < 0.5:
    image = tf.image.flip_left_right(image)
    
  if tf.random.uniform(()) < 0.5:
    image = tf.image.flip_up_down(image)

  return image

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)
    assert not FLAGS.batch_parse, 'image decode must before batch'

  def parse(self, example):
    self.auto_parse()
    f = self.parse_(serialized=example)

    f['image'] = decode_image(f['image'])
    if self.subset == 'train':
      f['image'] = augment(f['image'])

    mt.try_append_dim(f)

    x = f
    y = f['label']
    del f['label']

    return x, y
    
