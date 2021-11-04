#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige
#          \date   2021-01-09 17:51:06.853603
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gezi

import wandb
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import collections
from collections import OrderedDict, Counter, defaultdict
from gezi import logging, tqdm
from .config import *

from .modelDesign import NMSE

def evaluate(y_true, y_pred):
  nmse = NMSE(y_true, y_pred)
  recover = 1 - nmse
  res = {
    'recover': recover
  }
  res = gezi.dict_prefix(res, 'Metrics/')
  return res
