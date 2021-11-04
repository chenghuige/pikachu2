#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2021-07-31 08:49:47.444181
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
import numpy as np
from icecream import ic

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt as mt
from pretrain import config
from pretrain.config import *
from pretrain.dataset import Dataset

def main(_):
  timer = gezi.Timer()
  fit = mt.fit 

  # transformer = FLAGS.transformer.split('/')[-1]
  # FLAGS.model_dir = f'../working/pretrain/{transformer}'
  # FLAGS.epochs = 2
  # FLAGS.optimizer = 'bert-adamw'
  # FLAGS.lr = 5e-5
  # # FLAGS.batch_size = 1024
  # FLAGS.fp16 = True
  # FLAGS.gpus = -1

  config.init()
  mt.init()

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = mt.pretrain.bert.Model(FLAGS.transformer, 
                custom_model=FLAGS.custom_model, 
                embedding_path=FLAGS.embedding_path,
                vocab_size=FLAGS.vocab_size,
                hidden_size=FLAGS.hidden_size,
                num_attention_heads=FLAGS.num_attention_heads)

    ic(model.bert.layers)
    fit(model,  
        loss_fn=model.get_loss(),
        Dataset=Dataset,
        metrics=['accuracy']
       ) 

  model.bert.save_weights(f'{FLAGS.model_dir}/bert.h5')
  model.bert.save_pretrained(f'{FLAGS.model_dir}/bert')
  # model.dense.save(f'{FLAGS.model_dir}/dense.h5')
  # model.save(f'{FLAGS.model_dir}/model.h5')
  mt.save_dense(model.dense, f'{FLAGS.model_dir}/dense')

if __name__ == '__main__':
  app.run(main)   
