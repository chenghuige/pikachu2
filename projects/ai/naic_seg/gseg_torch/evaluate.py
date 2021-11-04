#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2020-10-11 13:04:05.805775
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import traceback

import torch

import gezi
logging = gezi.logging
from gezi import tqdm
from gezi.metrics.image.semantic_seg import Evaluator

from lele import to_torch
from gseg.config import *

def eval(dataset, model, eval_step, steps, step, is_last, num_examples, loss_fn, outdir):
  key_metric = 'FWIoU'
  evaluator = Evaluator(FLAGS.CLASSES)
  res = {}

  t = tqdm(enumerate(dataset), total=steps, ascii=True, desc= 'eval_loop')
  try:
    for step_, (x, y) in t:
      if steps and step_ == steps:
        break 
      x, y = to_torch(x, y)
      y_ = model(x)

      y = y.detach().cpu().numpy()
      if len(y.shape) == len(y_.shape):
        y = y.squeeze(-1)
      y_ = torch.argmax(y_, dim=1).detach().cpu().numpy()
      res_ = evaluator.eval(y, y_)

      t.set_postfix({key_metric: res_[key_metric]})
    try:
      res.update(evaluator.eval_once())
    except Exception:
      res['abc'] = 0.
      pass
  except Exception as e:
    logging.warning(traceback.format_exc())
    logging.warning('steps', steps)

  return res


def get_eval_fn():
  return eval
