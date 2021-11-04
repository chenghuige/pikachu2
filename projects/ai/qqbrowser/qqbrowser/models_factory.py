#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   models_factory.py
#        \author   chenghuige  
#          \date   2021-08-25 20:52:36.531808
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import gezi
import qqbrowser
from qqbrowser.models import *

def get_model(model_name='baseline'):
  module_ = getattr(qqbrowser.models, model_name)   
  if model_name == 'baseline':
    model = module_.MultiModal()
  else:
    model = module_.Model()
  ic(model_name, model)
  return model
    