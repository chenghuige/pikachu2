#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
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

import torch
from torch import nn
from torch.nn import functional as F

from wechat.dataset import Dataset
import wechat.eval as ev
from wechat import config
from wechat.config import *
from wechat.util import *

from wechat.pyt.util import get_optimizer
from wechat.pyt.loss import Criterion

from wechat.pyt.model import *
import wechat.pyt.model as base

def main(_):
  timer = gezi.Timer()
  FLAGS.torch = True
  config.init()
  melt.init()
  fit = melt.fit
 
  model = getattr(base, FLAGS.model)() 
  fit(model,  
      loss_fn=Criterion(),
      Dataset=Dataset,
      eval_fn=ev.evaluate,
      eval_keys=eval_keys,
      valid_write_fn=ev.valid_write,
      infer_write_fn=ev.infer_write,
      ) 

  if FLAGS.mode == 'test':
    elapsed = timer.elapsed_ms()
    info = gezi.get('info')
    num_examples = info['num_test_examples']
    num_objs = len(FLAGS.action_list)
    logging.info('num_examples', num_examples, 'num_objs', num_objs, '2000 insts mean ms per obj', (elapsed * 2000) / (num_examples * num_objs))


if __name__ == '__main__':
  app.run(main)  
  
