#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dump-lang.py
#        \author   chenghuige  
#          \date   2020-04-22 06:09:39.386953
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('/home/gezi/mine/pikachu/projects/bert/src')
import os

import sys
from config import * 

import melt
melt.init_flags()
FLAGS = melt.get_flags()
FLAGS.pretrained = '/home/gezi/data/huggingface/tf-xlm-roberta-base'

#from model import xlm_model as Model
from model import XlmModel as Model
model = Model()

print(model.layers)

idir = sys.argv[1]
model.load_weights(idir + '/model_weight.h5')

model.transformer.save_weights(idir + '/lang_weight.h5')


