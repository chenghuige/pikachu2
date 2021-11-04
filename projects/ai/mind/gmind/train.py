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
sys.path.append('..')
import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt as mt

from gmind.model import get_model
from gmind.dataset import Dataset
import gmind.evaluate as ev
from gmind import config
from gmind.config import *
from gmind.util import *
from gmind.loss import get_loss

def main(_):
  fit = mt.fit  
  config.init()
  mt.init()
  eval_keys=['uid', 'did', 'impression_id', 'uid_in_train', 'did_in_train', 'position', 'hist_len'] # used in evaluate.py for x 
  gezi.set('test_keys', ['impression_id', 'position']) # test only use impression_id and position

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = get_model(FLAGS.model)

    # 如果用model.get_model 当前不能用model.get_loss 否则tf2 keras
    # Inputs to eager execution function cannot be Keras symbolic tensors, but found [<tf.Tensor 'Squeeze_6:0' shape=(None,) dtype=int64>, <tf.Tensor 'Squeeze_10:0' shape=(None,) dtype=int64>]    
    # 当前存在问题 tf2.4 下面 functional model 不能save .h5 报错  Unable to create link (name already exists) 但是tf 2.3 可以 save_weights, save graph则都不可以
    # functional model 可以save checkpoint 而mt.Model是不可以的 但是mt.Model也就是非functional model可以save_weights 包括tf2.4 但是不能save checkpoint
    if not FLAGS.lm_target:
      callbacks = []
      infer_writer = InferWriter()
      fit(model,  
          loss_fn=get_loss(model),
          Dataset=Dataset,
          eval_fn=ev.evaluate,
          eval_keys=eval_keys,
          # out_hook=out_hook,
          infer_write_fn=infer_write,
          # infer_write_fn=infer_writer.write,
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

