#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2020-09-28 16:10:12.412785
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
sys.path.append('..')
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras
import gezi
import melt as mt
from gseg import config
from gseg.config import *
from gseg.dataset import Dataset
from gseg.evaluate import get_eval_fn
from gseg.util import get_infer_fn
from gseg.model import get_model
from gseg.loss import get_loss_fn
from gseg.metrics import get_metrics

def main(_):
  config.init()

  # 必须放到最前
  mt.init()

  # TODO FIXME目前最大的问题是多gpu情况下不能使用机器全部gpu 否则就hang 比如8卡机器 4，6没问题 但是不能8 6卡机器4卡没问题不能跑6， 2卡机器就只能单卡跑了...
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = get_model(FLAGS.model)
    loss_fn=model.get_loss() if hasattr(model, 'get_loss') else get_loss_fn()
    mt.fit(model, 
           loss_fn=loss_fn,
           Dataset=Dataset,
           metrics=get_metrics(),
           eval_fn=get_eval_fn(),
           inference_fn=get_infer_fn(),
          )

if __name__ == '__main__':
  app.run(main)  
