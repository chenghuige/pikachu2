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
import melt as mt

import projects.ai.mind.src.model as base
from projects.ai.mind.src.dataset import Dataset
from projects.ai.mind.src import loss
import projects.ai.mind.src.evaluate as ev
from projects.ai.mind.src import config
from projects.ai.mind.src.config import *
from projects.ai.mind.src.util import *

def main(_):
  fit = mt.fit  
  config.init()
  mt.init()
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = getattr(base, FLAGS.model)() 
    loss_fn = model.get_loss()

    # 如果用model.get_model 当前不能用model.get_loss 否则tf2 keras
    # Inputs to eager execution function cannot be Keras symbolic tensors, but found [<tf.Tensor 'Squeeze_6:0' shape=(None,) dtype=int64>, <tf.Tensor 'Squeeze_10:0' shape=(None,) dtype=int64>]
    if not FLAGS.custom_loss:
      loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # if tf.__version__ >= '2':
    #   # TODO FIXME has bug for tf1 non eager mode below but eager mode ok 
    #   # TF2 may has problem eager input not ok... 
    #   dataset = Dataset('train')
    #   dataset.make_batch()
    #   if not dataset.has_varlen_keys:
    #     inputs = dataset.get_inputs()
    #     model = model.get_model(inputs)
    
  if not FLAGS.lm_target:
    callbacks = []
    #if tf.__version__ >= '2':
    #  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir, profile_batch=(5, 20))
    #  callbacks = [tensorboard_callback]
    fit(model,  
        loss_fn=loss_fn,
        Dataset=Dataset,
        eval_fn=ev.evaluate,
        eval_keys=['uid_', 'did_', 'impression_id', 'uid_in_train', 'did_in_train'], # used in evaluate.py for x 
        out_hook=out_hook,
        infer_write_fn=infer_write,
        valid_write_fn=valid_write,
        callbacks=callbacks,
        ) 
  else:
    fit(model,  
        loss_fn=loss_fn,
        Dataset=Dataset,
        ) 

if __name__ == '__main__':
  app.run(main)  
