#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics.py
#        \author   chenghuige  
#          \date   2020-10-07 07:45:56.062319
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import melt as mt
from .config import *
from .util import *

def get_metrics():
  if is_classifier():
    return [mt.metrics.CategoryMIoU('IMAGE/CLASS/MIoU')]
  return mt.metrics.SemanticSeg(FLAGS.NUM_CLASSES).get_metrics()
