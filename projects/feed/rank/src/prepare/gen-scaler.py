#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-scaler.py
#        \author   chenghuige  
#          \date   2019-09-12 14:51:51.176799
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from tfrecord_lite import decode_example 
import gezi

import sys 
import os
import math
import pickle
from tqdm import tqdm

def main(_):
  if len(sys.argv) > 2:
    infile = sys.argv[1]
    out_dir = sys.argv[2]
  else:
    out_dir = sys.argv[1]
    infile = gezi.list_files(os.path.join(out_dir, 'tfrecord/train'))[0]

  total = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(infile))

  durs = []
  for it in tqdm(tf.compat.v1.python_io.tf_record_iterator(infile), total=total):
    x = decode_example(it)
    durs.append(x['duration'][0])
  durs_log = np.asarray([math.log(float(x + 1)) for x in durs if x > 0])

  ofile = os.path.join(out_dir, 'qt.normal.pkl')
  qt = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution='normal')
  qt.fit(durs_log.reshape([-1, 1]))
  pickle.dump(qt, open(ofile, 'wb'))

  ofile = os.path.join(out_dir, 'qt.uniform.pkl')
  qt = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution='uniform')
  qt.fit(durs_log.reshape([-1, 1]))
  pickle.dump(qt, open(ofile, 'wb'))


if __name__ == '__main__':
  tf.compat.v1.app.run()  
  
