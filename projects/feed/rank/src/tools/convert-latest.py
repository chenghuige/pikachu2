#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   convert-latest.py
#        \author   chenghuige  
#          \date   2019-08-27 16:16:31.920375
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import melt

model_dir = sys.argv[1]
model_path = melt.latest_checkpoint(model_dir)

command ='python ./tools/convert.py {} {}/model.bin'.format(model_path, model_dir) 
print(command)
os.system(command)

