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

from pyt.model import *
import pyt.model as base
import evaluate as ev

import melt
import gezi
import lele

from projects.feed.rank.src import config
from projects.feed.rank.src import util

from pyt.util import get_optimizer
from pyt.loss import Criterion

def main(_):
  FLAGS.torch = True
  config.init()
  melt.init()
  fit = melt.fit
 
  Dataset = util.prepare_dataset()

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  # loss_fn = nn.BCEWithLogitsLoss()
  loss_fn = Criterion()

  eval_fn, eval_keys = util.get_eval_fn_and_keys()
  valid_write_fn = ev.valid_write
  out_hook = ev.out_hook

  weights = None if not FLAGS.use_weight else 'weight'

  fit(model,  
      loss_fn,
      Dataset,
      optimizer=get_optimizer,
      eval_fn=eval_fn,
      eval_keys=eval_keys,
      valid_write_fn=valid_write_fn,
      out_hook=out_hook,
      weights=weights)

if __name__ == '__main__':
  app.run(main)  
  
