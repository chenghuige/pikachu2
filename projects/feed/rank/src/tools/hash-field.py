#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   check-conflict.py
#        \author   chenghuige  
#          \date   2019-10-26 14:00:08.584195
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import gezi

print(gezi.hash_int64(sys.argv[1]) % 10000)
