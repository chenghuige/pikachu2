#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2018-12-07 15:19:16.505094
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
import pandas as pd
from sklearn import metrics
import gezi

from config import *

def evaluate(labels, logits):
  logits = logits[:, :NUM_CLASSES]
  predicts = np.argmax(logits, -1)
  acc = np.mean(np.equal(predicts, labels))

  probs = gezi.softmax(logits)
  loss = metrics.log_loss(labels, probs)

  kappa = metrics.cohen_kappa_score(labels, predicts)

  result = dict(loss=loss, acc=acc, kappa=kappa)

  return result

def write(ids, labels, logits, ofile):
  logits = logits[:, :NUM_CLASSES]
  df = pd.DataFrame()
  df['id_code'] = ids
  predicts = np.argmax(logits, -1)
  df['diagnosis'] = predicts
  if labels is not None: 
    df['label'] = labels
  df = df.sort_values('id_code') 
  df.to_csv(ofile, index=False)

def valid_write(ids, labels, logits, ofile):
  return write(ids, labels, logits, ofile)

def infer_write(ids, logits, ofile):
  return write(ids, None, logits, ofile)
