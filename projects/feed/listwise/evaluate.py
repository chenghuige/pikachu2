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

# https://github.com/qiaoguan/deep-ctr-prediction/blob/master/DeepCross/metric.py
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np
# import multiprocessing
# from multiprocessing import Manager
# import pymp
from tqdm import tqdm
from scipy.stats import weightedtau, kendalltau
from scipy.stats._stats import _kendall_dis
import math

import gezi
logging = gezi.logging 

def evaluate(y_true, y_pred, x, other):
  print(y_true)
  print(y_pred)
  print(x['durs'])