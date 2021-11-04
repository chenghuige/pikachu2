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
import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt as mt

import mrc_yesno as myn
from mrc_yesno.model import get_model
from mrc_yesno.dataset import Dataset
import mrc_yesno.eval as ev
from mrc_yesno import config
from mrc_yesno.config import *
from mrc_yesno.util import *
from mrc_yesno.loss import get_loss

def main(_):
  fit = mt.fit  
  config.init()
  mt.init()
  eval_keys=['id'] # used in evaluate.py for x 

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = get_model(FLAGS.model)
    fit(model,  
        loss_fn=get_loss(model),
        Dataset=Dataset,
        eval_fn=ev.evaluate,
        eval_keys=eval_keys,
        # valid_write_fn=valid_write,
        ) 


if __name__ == '__main__':
  app.run(main)  