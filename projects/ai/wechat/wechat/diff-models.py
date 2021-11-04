#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-tfrecord.py
#        \author   chenghuige  
#          \date   2019-09-11 11:00:01.818073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
from collections import OrderedDict
import pandas as pd
import melt as mt
import gezi
from gezi import tqdm

from wechat.config import *

def main(_):
  with gezi.Timer('read csv', print_fn=ic):
    d1 = pd.read_csv(sys.argv[1])
    d2 = pd.read_csv(sys.argv[2])

  res = OrderedDict()
  for action in tqdm(ACTIONS, desc='actions'):
    diff = gezi.metrics.inverse_ratio(d1[action].values, d2[action].values)
    res[action] = diff
  
  ic(res)

if __name__ == '__main__':
  app.run(main)  
  
