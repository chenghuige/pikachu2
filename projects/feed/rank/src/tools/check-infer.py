#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   check-infer.py
#        \author   chenghuige  
#          \date   2020-01-09 14:56:22.265390
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import traceback
import pandas as pd
 
try:
  df = pd.read_csv(sys.argv[1])
  df = df[df.abtest==45600]
  auc = df.auc[0]
except Exception:
  auc = -1.
print(auc)

