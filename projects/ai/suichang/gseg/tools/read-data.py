#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read_data.py
#        \author   chenghuige  
#          \date   2020-09-27 02:24:41.728073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('..')
import os
import gezi
from gezi import tqdm
import melt
melt.init_flags()
from melt import FLAGS

from gseg.dataset import Dataset

FLAGS.batch_parse = False

files = gezi.list_files('../input/tfrecords/train/0/*')
#ds = Dataset('valid').make_batch(32, files, repeat=False)
ds = Dataset('valid', files=files)
print(len(ds))
print(ds.num_instances)
d = ds.make_batch(32, repeat=False)
num_steps = ds.num_steps
#x, y = next(iter(d))
#print(x)
#print(y)
#print(x.keys(), x['image'].shape, y.shape)

for x in tqdm(d, total=num_steps):
  pass

for x in tqdm(d, total=num_steps):
  pass

for x in tqdm(d, total=num_steps):
  pass
