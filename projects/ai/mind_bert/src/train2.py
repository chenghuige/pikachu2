#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2020-04-12 20:14:09.102032
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging

import melt
import config
from config import *
import model as base
import loss
from evaluate import evaluate
from dataset import Dataset

from husky.callbacks import EvalCallback

def main(_):
  config.init()
  fit = melt.fit
  melt.init()

  strategy = melt.distributed.get_strategy()
  with strategy.scope():
    model = getattr(base, FLAGS.model)() 
    loss_fn = loss.get_loss_fn()
 
  FLAGS.lr=3e-5
  fit(model,  
      loss_fn,
      Dataset,
      eval_fn=evaluate,
      eval_keys=['lang']
      )   
  
  FLAGS.lr=1.5e-5
  fit(model,  
      loss_fn,
      Dataset,
      eval_fn=evaluate,
      eval_keys=['lang']
      )   


if __name__ == '__main__':
  app.run(main)  

  
