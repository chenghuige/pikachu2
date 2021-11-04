#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tflite.py
#        \author   chenghuige  
#          \date   2020-10-28 06:14:07.941399
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import pathlib
import tensorflow as tf

keras_model_file = sys.argv[1]
tflite_model_file = keras_model_file.replace('.h5', '.tflite')
if len(sys.argv) > 2: 
  tflite_model_file = sys.argv[2]
print(f'Start converting {keras_model_file} to {tflite_model_file}')
keras_model = tf.keras.models.load_model(keras_model_file, custom_objects={'tf': tf}, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
pathlib.Path(tflite_model_file).write_bytes(tflite_model)
print(f'Done converting {keras_model_file} to {tflite_model_file}')
