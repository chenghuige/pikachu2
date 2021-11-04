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
import melt

import model as base
from dataset import Dataset
import evaluate as ev
import loss
# from config import *
from projects.ai.ad.src.config import *
from util import *

def main(_):
  fit = melt.fit  
  if FLAGS.lm_target:
    FLAGS.fold = None
    FLAGS.train_input='../input/tfrecords/train,../input/tfrecords/test,../input/tfrecords/valid'

  melt.init()
  
  strategy = melt.distributed.get_strategy()
  with strategy.scope():
    model = getattr(base, FLAGS.model)() 
    gezi.set('model', model)
    loss_fn = getattr(loss, FLAGS.loss) 

  if not FLAGS.lm_target:
    fit(model,  
        loss_fn=loss_fn,
        Dataset=Dataset,
        eval_fn=ev.evaluate,
        eval_keys=['id', 'gender', 'age'],
        out_hook=out_hook,
        infer_write_fn=infer_write,
        ) 
  else:
    fit(model,  
        loss_fn=melt.losses.sampled_bilm_loss,
        Dataset=Dataset,
        ) 

if __name__ == '__main__':
  app.run(main)  
