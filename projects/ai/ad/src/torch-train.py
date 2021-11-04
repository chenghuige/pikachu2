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
import os

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.nn import functional as F

import melt
import gezi
import lele

import pyt.model as base
from dataset import Dataset
import evaluate as ev
from projects.ai.ad.src.config import *
from util import *
from pyt.loss import Criterion

def main(_):
  FLAGS.torch = True
  # config.init()
  FLAGS.optimizer = 'bert-Adam'
  FLAGS.write_valid = False
  melt.init()
  fit = melt.fit
 
  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  loss_fn =  Criterion()

  fit(model,  
      loss_fn=loss_fn,
      Dataset=Dataset,
      eval_fn=ev.evaluate,
      out_hook=out_hook,
      eval_keys=['id', 'gender', 'age'],
      ) 

if __name__ == '__main__':
  app.run(main)  
  
