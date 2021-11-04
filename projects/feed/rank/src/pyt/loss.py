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
  
  def calc_loss(self, y_,  y, x, model, weights=None):
    # bloss_fn = nn.BCEWithLogitsLoss()
    bloss_fn = nn.BCEWithLogitsLoss(weights)
    if FLAGS.num_gpus <= 1:
      # TODO DataParallel can not support model.attr, notice for distributed FLAGS.num_gpus==0
      loss = bloss_fn(model.logit, y)
    else:
      loss = bloss_fn(y_, y)
    return loss
  
  def calc_multi_loss(self, y_,  y, x, model, weights=None):
    assert FLAGS.num_gpus <= 1
    
    duration = x['duration'].float()
    dur_unknown = (duration < 0).float()
    duration = torch.clamp(duration, max=float(FLAGS.max_duration))
    dur_prob = duration / float(FLAGS.max_duration)
    
    click_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    dur_loss_fn = getattr(nn, FLAGS.multi_obj_duration_loss)(reduction='none')
    
    y_ = model.y_click
    click_loss = click_loss_fn(y_, y)
    
    y_ = model.y_dur
    dur_loss = dur_loss_fn(y_, dur_prob) * (1. - dur_unknown)
    if FLAGS.use_jump_loss:
      dur_loss *= y
      
    ratio = FLAGS.multi_obj_duration_ratio 
    loss = click_loss * (1. - ratio) + dur_loss * ratio 
    
    sample_weight = x['weight']
    loss =(loss * sample_weight).sum() / sample_weight.sum()
    return loss

  
  def __call__(self, y_,  y, x, model, weights=None):
    if not FLAGS.multi_obj_type:
      return self.calc_loss(y_, y, x, model, weights)
    else:
      return self.calc_multi_loss(y_, y, x, model, weights)
