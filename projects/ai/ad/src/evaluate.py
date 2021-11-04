#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2020-05-24 12:17:53.108984
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
from tqdm import tqdm
import gezi
logging = gezi.logging 
import melt

from util import *
from projects.ai.ad.src.config import *

step = 1
def evaluate(y_true, y_pred, x, other):
  global step
  gender = gezi.squeeze(x['gender'])
  age = gezi.squeeze(x['age'])

  gender_ = other['pred_gender']
  age_ = other['pred_age']

  logging.debug(list(zip(age, age_))[:30])
  logging.debug(list(zip(gender, gender_))[:30])

  age = np.asarray([to_age(x) for x in age])
  age_ = np.asarray([to_age(x)  for x in age_])

  gender = np.asarray([to_gender(x) for x in gender])
  gender_ = np.asarray([to_gender(x) for x in gender_])
  
  acc_gender = np.sum(gender == gender_) / len(gender)
  acc_age = np.sum(age == age_) / len(age)

  acc_all = acc_gender + acc_age

  logging.debug(list(zip(age, age_))[:30])
  logging.debug(list(zip(gender, gender_))[:30])

  logger = melt.get_summary_writer()
  if FLAGS.ev_first:
    step -= 1
  confusion_gender = gezi.confusion(gender, gender_, info='acc/gender:{:.4f}'.format(acc_gender))
  logger.image('confusion/gender', confusion_gender, step, bytes_input=True)
  confusion_age = gezi.confusion(age, age_, info='acc/age:{:.4f}'.format(acc_age))
  logger.image('confusion/age', confusion_age, step, bytes_input=True)

  step += 1
  return {'acc/all': acc_all, 'acc/age': acc_age, 'acc/gender': acc_gender}
