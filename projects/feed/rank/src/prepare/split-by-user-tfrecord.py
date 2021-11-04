#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   split-by-user.py
#        \author   chenghuige  
#          \date   2019-08-19 09:54:20.684307
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
import melt
import gezi
from tfrecord_lite import decode_example 
import tensorflow as tf
from absl import app
import multiprocessing 
# tf.enable_eager_execution()

in_dir = sys.argv[1]
files = gezi.list_files(in_dir)

user_infos = sys.argv[3]
out_dir = sys.argv[2]

total_parts = 8
if len(sys.argv) > 4:
  total_parts = int(sys.argv[4])

if not os.path.exists(out_dir):
  os.system('mkdir -p %s' % out_dir)

out_train = os.path.join(out_dir, 'train')
out_valid = os.path.join(out_dir, 'valid')

users = [line.rstrip().split('\t')[0] for line in open(user_infos)]
users = np.asarray(users)
np.random.shuffle(users)

num_users = len(users)
num_valid_users = int(len(users) / total_parts)

valid_users = set(users[:num_valid_users])
train_users = set(users[num_valid_users:])

def build_features(file):
  file_name = os.path.basename(file)
  print(out_train)
  print(file_name)
  ofile_train = os.path.join(out_train, file_name)
  ofile_valid = os.path.join(out_valid, file_name)
  print(ofile_train)
  print(ofile_valid)
  example = tf.train.Example()
  num_records = melt.get_num_records([file])
  with melt.tfrecords.Writer(ofile_train) as writer_train, melt.tfrecords.Writer(ofile_valid) as writer_valid:
    for record in tqdm(tf.io.tf_record_iterator(file), total=num_records):
    # for record in tf.data.TFRecordDataset(file):
      example.ParseFromString(record)
      f = example.features.feature
      uid = f['id'].bytes_list.value[0].decode().split('\t')[0]
      if uid in train_users:
        writer_train.write(example)
      else:
        writer_valid.write(example)
      
      
def main(_):
  pool = multiprocessing.Pool()

  pool.map(build_features, files)
  pool.close()
  pool.join()

if __name__ == '__main__':
  app.run(main) 