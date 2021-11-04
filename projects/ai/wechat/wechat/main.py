#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
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

from wechat.model import get_model
from wechat.dataset import Dataset
import wechat.eval as ev
from wechat import config
from wechat.config import *
from wechat.util import *
from wechat.loss import get_loss

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  config.init()
  mt.init()
  
  if FLAGS.online and 'tione' in os.environ['PATH']:
    team_config_file = '/home/tione/notebook/team_config.json'
    if os.path.exists(team_config_file):
      os.system(f'cp {team_config_file} {FLAGS.model_dir}')
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = get_model(FLAGS.model)
    fit(model,  
        loss_fn=get_loss(model),
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
