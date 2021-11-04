#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2019-07-27 22:33:36.314010
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('..')
import os

from absl import app, flags
FLAGS = flags.FLAGS

import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
import scipy.io as sio

import gezi
from gezi import pad
import melt

import tensorflow as tf
data = None

def build_features(index):
  start, end = gezi.get_fold(len(data), FLAGS.num_records, index)
  data_ = data[start: end]
  ofile = f'{FLAGS.out_dir}/{FLAGS.mark}/record_{index}.tfrec'
  total = len(data_)
  with melt.tfrecords.Writer(ofile) as writer:
    for instance in tqdm(data_, total=total):
      feature = {}
      feature['data'] = instance
      writer.write_feature(feature)

def main(_):
  np.random.seed(FLAGS.seed_)

  FLAGS.out_dir = f'{FLAGS.out_dir}/{FLAGS.record_name}'
  gezi.try_mkdir(f'{FLAGS.out_dir}/{FLAGS.mark}')

  global data
  mat = sio.loadmat(f'{FLAGS.in_dir}/{FLAGS.ifile}')
  data = mat['H_4T4R']
  data = data.astype('float32')
  CHANNEL_SHAPE_DIM1 = 24
  CHANNEL_SHAPE_DIM2 = 16
  CHANNEL_SHAPE_DIM3 = 2
  data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1 * CHANNEL_SHAPE_DIM2 * CHANNEL_SHAPE_DIM3))

  # build_features(0)
  with Pool(FLAGS.num_records) as p:
    p.map(build_features, range(FLAGS.num_records))

if __name__ == '__main__':
  flags.DEFINE_string('ifile', 'H_4T4R.mat', '')
  flags.DEFINE_string('in_dir', '../input', '')
  flags.DEFINE_string('out_dir', '../input', '')
  flags.DEFINE_string('mark', 'train', 'train or dev')
  flags.DEFINE_integer('num_records', 20, '')
  flags.DEFINE_integer('seed_', 12345, '')
  flags.DEFINE_string('record_name', 'tfrecords', '')
  flags.DEFINE_integer('max_len', 512, '')
  
  app.run(main) 
