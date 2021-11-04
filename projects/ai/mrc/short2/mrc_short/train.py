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

import mrc_short as myn
from mrc_short.model import get_model
from mrc_short.dataset import Dataset
import mrc_short.eval as ev
from mrc_short import config
from mrc_short.config import *
from mrc_short.util import *
from mrc_short.loss import get_loss

def main(_):
  fit = mt.fit  
  config.init()
  mt.init()
  eval_keys=['qid', 'pid', 'passage_has_answer', 'start_position', 'end_position'] # used in evaluate.py for x 

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = get_model(FLAGS.model)
    fit(model,  
        loss_fn=get_loss(model),
        Dataset=Dataset,
        eval_fn=ev.evaluate,
        eval_keys=eval_keys,
        valid_write_fn=ev.valid_write,
        ) 


if __name__ == '__main__':
  app.run(main)  
