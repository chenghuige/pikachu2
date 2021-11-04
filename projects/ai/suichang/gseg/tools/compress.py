#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   compress.py
#        \author   chenghuige  
#          \date   2020-10-28 10:23:33.253532
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
import tensorflow_model_optimization as tfmot

keras_model_file = sys.argv[1]
compressed_model_file = keras_model_file.replace('.h5', '.compress.h5')
if len(sys.argv) > 2: 
  compressed_model_file = sys.argv[2]
print(f'Start converting {keras_model_file} to {compressed_model_file}')

keras_model = tf.keras.models.load_model(keras_model_file, custom_objects={'tf': tf}, compile=False)

compressed_model = tfmot.sparsity.keras.strip_pruning(keras_model)

tf.keras.models.save_model(compressed_model, compressed_model_file, include_optimizer=False)
print(f'Done converting {keras_model_file} to {compressed_model_file}')
