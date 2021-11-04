#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval-all.py
#        \author   chenghuige
#          \date   2021-08-06 11:17:22.595222
#   \Description
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
from gezi import tqdm

files = glob.glob(f'{sys.argv[1]}/*/valid.csv')
files = [x for x in files if not os.path.exists(x.replace('valid.csv', 'metrics.csv.bak'))]
ic(files[:5])
for file_ in tqdm(files):
  command = f'./eval.py --efs {file_}'
  ic(command)
  os.system(command)

