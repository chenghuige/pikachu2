#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2020-05-28 16:18:08.729010
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import pandas as pd
from projects.ai.ad.src.config import *

# TODO FIXME why /home/gezi/mine/pikachu/utils/husky/callbacks/evaluate.py turn to str...
def to_gender(x):
  return int(int(x) > FLAGS.gender_thre) + 1

# def to_age(x, intervals=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]):
#   if intervals is None:
#     return int(x * 10) + 1
#   else:
#     for i, interval in enumerate(intervals):
#       if x <= interval:
#         return i + 1
#     return i + 1

def to_age(x):
  if 'Cls' in FLAGS.model:
    return int(x) + 1
  else:
    return int(x * 10) + 1

def out_hook(model): 
  return dict(pred_gender=model.pred_gender, pred_age=model.pred_age)
 
def infer_write(ids, predicts, ofile, others):
  pred_age = others['pred_age']
  pred_gender = others['pred_gender']
  pred_age = np.asarray([to_age(x) for x in pred_age])
  pred_gender = np.asarray([to_gender(x)  for x in pred_gender])
  df = pd.DataFrame({'user_id': ids, 'predicted_age': pred_age, 'predicted_gender': pred_gender})
  df.to_csv(ofile, index=False)
