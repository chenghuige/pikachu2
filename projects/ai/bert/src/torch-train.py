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

import torch
from torch import nn
from torch.nn import functional as F

import melt
import gezi
import lele

import gezi
logging = gezi.logging

from pyt.model import *
import pyt.model as base
import config
from config import *
from evaluate import evaluate
from dataset import Dataset

def main(_):
  FLAGS.torch = True
  config.init()
  fit = melt.fit
  melt.init()

  model_name = FLAGS.model
  model = getattr(base, model_name)() 
  loss_fn = nn.BCELoss()

  fit(model,  
      loss_fn,
      Dataset,
      eval_fn=evaluate,
      eval_keys=['lang']
      )   

if __name__ == '__main__':
  app.run(main)  

  
