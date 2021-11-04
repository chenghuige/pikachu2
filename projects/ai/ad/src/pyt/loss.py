#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2018-09-17 20:34:23.281520
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np

from absl import flags
FLAGS = flags.FLAGS

import lele
import torch
from torch import nn
from torch.nn import functional as F

class Criterion(object):
  def __init__(self):
    pass
  
  def __call__(self, y_,  y, x, model):
    bloss_fn = nn.BCEWithLogitsLoss()
    loss_gender = bloss_fn(model.gender, x['gender'].type_as(model.gender))
    mloss_fn = nn.CrossEntropyLoss()
    loss_age = mloss_fn(model.age, x['age'].squeeze())
    return loss_age + loss_gender
