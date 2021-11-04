#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   avg-ensemble.py
#        \author   chenghuige  
#          \date   2019-08-27 23:38:36.572572
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd 

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

df1 = df1.sort_values('id') 
df2 = df2.sort_values('id')

