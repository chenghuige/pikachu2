#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   get-all-uers.py
#        \author   chenghuige  
#          \date   2019-08-18 11:06:39.496266
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import collections
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import traceback
import pandas as pd

import melt
import gezi
logging = gezi.logging
import tensorflow as tf
import pickle

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('ofile', None, '')
flags.DEFINE_bool('title', False, '')

from projects.feed.rank.src.tfrecord_dataset import Dataset

#  python loop-save.py /search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520,/search/odin/publicData/CloudS/libowei/rank4/newmse/data/video_hour_newmse_v1/tfrecords/2020051520,/search/odin/publicData/CloudS/libowei/rank4/shida/data/video_hour_shida_v1/tfrecords/2020051520

def main(_):  
  FLAGS.batch_size = FLAGS.batch_size_
  batch_size = FLAGS.batch_size

  with gezi.Timer('load--', print_fn=print, print_before=True) as timer:
    f = open('/home/gezi/data/rank/video.npy', 'rb')
    m = pickle.load(f)
  dataset = tf.data.Dataset.from_tensor_slices(m)
  dataset = dataset.shuffle(FLAGS.batch_size * 100).batch(FLAGS.batch_size)

  for x in dataset:
    print(x)
    break
	
if __name__ == '__main__':
  app.run(main)
  
