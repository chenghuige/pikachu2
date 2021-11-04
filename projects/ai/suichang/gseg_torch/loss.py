#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2020-10-11 13:04:28.110895
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch
from torch import nn
  
def get_loss_fn():
  def calc_loss(y_pred, y_true, x=None, model=None):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(y_pred, y_true.squeeze(-1).to(dtype=torch.long))
  return calc_loss
