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
import os

from absl import app as absl_app
from absl import flags
FLAGS = flags.FLAGS

import glob
from tqdm import tqdm
import multiprocessing
from multiprocessing import Value

import gezi
import melt

from config import *
import pandas as pd

import tensorflow as tf

flags.DEFINE_string('input', None, '')
flags.DEFINE_string('out_dir', None, '')
flags.DEFINE_bool('over_write', True, '')


counter = Value('i', 0)

def get_out_file(infile):
  infile_ = os.path.basename(infile)
  ofile_ = infile_.replace('.csv', '.record')
  ofile = os.path.join(FLAGS.out_dir, ofile_)
  return ofile

def build_features(infile):
  ofile = get_out_file(infile)
  if not FLAGS.over_write:
    if os.path.exists(ofile):
      print('-----exists', ofile)
      return
  df = pd.read_csv(infile)
  with melt.tfrecords.Writer(ofile) as writer:
    for _, row in tqdm(df.iterrows(), total=len(df), ascii=True):
      id = row['id_code']
      label = 0
      if 'diagnosis' in row:
        label = row['diagnosis']
      image_file = f'{FLAGS.out_dir}_images/{id}.png'
      image = melt.read_image(image_file)
      feature = {'id': melt.bytes_feature(id),
                 'label': melt.int64_feature(label),
                 'image': melt.bytes_feature(image)}
                
      record = tf.train.Example(features=tf.train.Features(feature=feature))

      writer.write(record)
      global counter
      with counter.get_lock():
        counter.value += 1


def main(_):
  input = f'{FLAGS.dir}/{FLAGS.input}'
  files = glob.glob(input)
  gezi.sprint(input)
  
  FLAGS.out_dir = f'{FLAGS.dir}/tfrecords/{FLAGS.out_dir}'
  out_dir = FLAGS.out_dir
  if not os.path.exists(out_dir):
    print('make new dir: [%s]' % out_dir, file=sys.stderr)
    os.makedirs(out_dir)
  gezi.sprint(out_dir)

  pool = multiprocessing.Pool()
  pool.map(build_features, files)
  pool.close()
  pool.join()

  print('num_records:', counter.value)

  out_file = '{}/num_records.txt'.format(out_dir)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  absl_app.run(main) 
