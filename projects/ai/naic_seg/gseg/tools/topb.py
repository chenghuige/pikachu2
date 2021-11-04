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
from gseg.third.bonlime.model import *
model = mt.load_model(sys.argv[1])
# input = model.input
#input._name = 'image'
input = tf.keras.Input(shape=(256, 256, 3), name='images')
out = model(input, training=False)
out = tf.argmax(out, -1)
mask = tf.cast(out < 4, tf.int64)
out = (out + 1) * mask + (out + 3) * (1 - mask)
out = tf.expand_dims(out, -1)
out = tf.identity(out, name='pred')
model = tf.keras.models.Model(input, out) 
model.summary()
model_name = 'deepllab_v3_plus'
out_dir = f'../working/convert/{model_name}'
gezi.try_mkdir(out_dir)
model_name = 'model.pb'
convert_to_froze_graph(model, model_name, out_dir)
print(out_dir, model_name)
print(input, out)
