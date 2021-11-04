#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:02.802049
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
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt as mt

from imt2020.model import get_model
from imt2020.dataset import Dataset
import imt2020.eval as ev
from imt2020 import config
from imt2020.config import *
from imt2020.util import *
from imt2020.loss import get_loss

def main(_):
  fit = mt.fit  
  config.init()
  mt.init()

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model, encModel, decModel = get_model(FLAGS.model)
    fit(model,  
        loss_fn=get_loss(model),
        Dataset=Dataset,
        eval_fn=ev.evaluate,
        ) 

  encModel.save('../input/submit_tf/modelSubmit/encoder.h5')
  decModel.save('../input/submit_tf/modelSubmit/decoder.h5')


if __name__ == '__main__':
  app.run(main)  
