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
from w2v import config
from w2v.config import *
from w2v.dataset import Dataset
from w2v.model import *


def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  config.init()
  mt.init()

  vocab = vocabs[FLAGS.attr]
  word2vec = Word2Vec(vocab.size(), FLAGS.emb_dim, FLAGS.num_negs)

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    fit(word2vec,  
        loss_fn=word2vec.get_loss(),
        Dataset=Dataset,
        metrics=['accuracy']
       ) 

  weights = word2vec.get_layer('w2v_embedding').get_weights()[0].numpy()
  np.save(f'{FLAGS.model_dir}/{FLAGS.attr}_w2v_{FLAGS.window_size}.npy')
  

if __name__ == '__main__':
  app.run(main)   
