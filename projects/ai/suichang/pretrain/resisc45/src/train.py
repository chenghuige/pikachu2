#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2020-10-18 20:44:26.075353
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
from tensorflow.keras.layers import Input
from classification_models.tfkeras import Classifiers
import efficientnet.tfkeras as eff

import melt as mt
from dataset import Dataset

class ModelWrapper(mt.Model):
  def __init__(self, model, preprocess, **kwargs):
    super(ModelWrapper, self).__init__(**kwargs)
    self.model = model
    self.preprocess = preprocess
    n_classes = 45
    self.dense = keras.layers.Dense(n_classes, activation=None)

  def deal(self, x):
    x = self.preprocess(x)
    x = self.model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    y = self.dense(x)
    return y

  def call(self, x):
    x = x['image']
    y = self.deal(x)
    return y

  def get_model(self):
    img_input = Input(shape=(256, 256, 3), name='image')    
    # # id_input = Input(shape=(1,), name='id') 
    # # label_input = Input(shape=(1,), name='label')
    # # cat_input = Input(shape=(1,), name='cat')
    # inp = {
    #   'image': img_input, 
    #   # 'id': id_input,
    #   # 'label': label_input,
    #   # 'cat': cat_input
    #   }
    inp = img_input
    out = self.deal(inp)
    return keras.Model(inp, out, name=f'resisc45_{FLAGS.model}')

def main(_):
  mt.init()

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    if FLAGS.model.startswith('Eff'):
      Model, preprocess_input = getattr(eff, FLAGS.model), eff.preprocess_input
    else:
      Model, preprocess_input = Classifiers.get(FLAGS.model)
    base_model = Model(input_shape=(256, 256,3), weights='imagenet', include_top=False)
    model = ModelWrapper(base_model, preprocess_input)

    mt.fit(model, 
           loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           Dataset=Dataset,
           metrics=['acc'],
          )

  mt.save_model(model.get_model(), os.path.join(FLAGS.model_dir, 'model.h5'))
  mt.save_model(base_model, os.path.join(FLAGS.model_dir, 'model_notop.h5'))

if __name__ == '__main__':
  app.run(main)  