#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   freeze-pb.py
#        \author   chenghuige  
#          \date   2020-10-28 09:09:12.549688
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import pathlib
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

keras_model_file = sys.argv[1]
pb_model_file = keras_model_file.replace('.h5', '.pb')
if len(sys.argv) > 2: 
  pb_model_file = sys.argv[2]
print(f'Start converting {keras_model_file} to {pb_model_file}')

# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = './model'
model_fname = './model/model.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)

session = tf.keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)


# keras_model = tf.keras.models.load_model(keras_model_file, custom_objects={'tf': tf}, compile=False)
# converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_model = converter.convert()
# pathlib.Path(tflite_model_file).write_bytes(tflite_model)
# print(f'Done converting {keras_model_file} to {pb_model_file}') 
