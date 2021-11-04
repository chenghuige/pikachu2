#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   topb.py
#        \author   chenghuige  
#          \date   2020-12-17 08:30:44.816447
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.layers import Input

def convert_to_froze_graph(keras_model: tf.python.keras.models.Model, model_name: str,
                           output_folder: str):
    """
    Export keras model to frozen model.

    Args:
        keras_model (tensorflow.python.keras.models.Model):
        model_name (str): Model name for the file name.
        output_folder (str): Output folder for saving model.

    """
    full_model = tf.function(lambda x: keras_model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    print(f"Model inputs: {frozen_func.inputs}")
    print(f"Model outputs: {frozen_func.outputs}")

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=output_folder,
                      name=model_name,
                      as_text=False)

os.environ['SM_FRAMEWORK'] = 'tf.keras'
sys.path.append('..')
import melt as mt
import gezi
model = tf.saved_model.load('../working/v0/electra/saved_model')
# input = model.input
#input._name = 'image'
max_len = 300
inputs = {
            'input_ids': Input(shape=(max_len,), dtype=tf.int32, name="input_ids"),
            'input_mask': Input(shape=(max_len,), dtype=tf.int32, name="input_mask"),
            'segment_ids': Input(shape=(max_len,), dtype=tf.int32, name="segment_ids"),
         }
out = model(inputs, training=False)

model = tf.keras.models.Model(inputs, out) 
model.summary()
model_name = 'model.pb'
out_dir = '../working/v0/electra'
convert_to_froze_graph(model, model_name, out_dir)
print(out_dir, model_name)
print(input, out)
