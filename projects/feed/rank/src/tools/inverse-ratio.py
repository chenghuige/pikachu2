#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   inverse-ratio.py
#        \author   chenghuige  
#          \date   2019-11-08 12:02:37.425699
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
abid = int(sys.argv[2]) * 100

inv_rate = df[df.abtest==abid].inv_rate.values[0]
print('%.5f' % inv_rate)
