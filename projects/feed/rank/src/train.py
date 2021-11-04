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

import os
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging

from projects.feed.rank.src import config
from projects.feed.rank.src import util
from model import *
import model as base
import loss
import evaluate as ev

def main(_):
  config.init()
  fit = melt.fit

  loss.init()
  melt.init()

  Dataset = util.prepare_dataset()

  model_name = FLAGS.model
  strategy = melt.distributed.get_strategy()
  with strategy.scope():
    model = getattr(base, model_name)() 
    out_keys = ['prob_click', 'prob_dur', 'num_tw_histories', 'num_vd_histories']
    
    # example = next(iter(Dataset('train').make_batch(FLAGS.batch_size, gezi.list_files(FLAGS.train_input.split('|')[0]))))
    # model(example[0])
    # model.summary()

    loss_fn = model.get_loss()
    # print(loss_fn(example[1], model(example[0])))

    eval_fn, eval_keys = util.get_eval_fn_and_keys()
    # model.eval_keys = eval_keys
    valid_write_fn = ev.valid_write
    out_hook = ev.out_hook

    # for different vars with different optimizers, learning rates
    variables_list_fn = util.get_variables_list

    fit(model,  
        loss_fn,
        Dataset,
        variables_list_fn=variables_list_fn,
        eval_fn=eval_fn,
        eval_keys=eval_keys,
        out_keys=out_keys,
        valid_write_fn=valid_write_fn,
        )   

if __name__ == '__main__':
  app.run(main)  
