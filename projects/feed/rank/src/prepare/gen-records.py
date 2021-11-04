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
from multiprocessing import Value, Manager
counter = Value('i', 0)
flen_counter = Value('i', 0)
from collections import defaultdict
import numpy as np

import gezi
import melt
from text_dataset import Dataset
import config
from config import *

import tensorflow as tf

dataset = None
user_dict = {}

def get_out_file(infile):
  infile_ = os.path.basename(infile)
  ofile_ = infile_ + '.record'
  ofile = os.path.join(FLAGS.out_dir, ofile_)
  return ofile

def build_features(infile):
  ofile = get_out_file(infile)
  if not FLAGS.over_write:
    if os.path.exists(ofile):
      print('-----exists', ofile)
      return
  print('----------writing to', ofile)
  mask_fields = set(map(int, FLAGS.mask_fields.split(','))) if FLAGS.mask_fields else set()

  total = sum(1 for _ in open(infile))
  with melt.tfrecords.Writer(ofile) as writer:
    for line in tqdm(open(infile), total=total, ascii=True):
      fields = line.rstrip().split('\t')
      num_infos = 4
      if len(fields) > num_infos:
        click = int(fields[0])
        if ':' not in fields[3]:
          duration = int(fields[1])
          if duration > 60 * 60 * 12:
            duration = 60
          id = '{}\t{}'.format(fields[2], fields[3])
          uid = fields[2]
        else:
          id = '{}\t{}'.format(fields[1], fields[2])
          uid = fields[1]
          if click:
            duration = 20 
          else:
            duration = 0
          num_infos = 3
        if FLAGS.has_emb:
          feat_id, feat_field, feat_value, cycle_profile_click, cycle_profile_show, cycle_profile_dur = dataset.get_feat_portrait(fields[num_infos:])
        else:
          feat_id, feat_field, feat_value = dataset.get_feat(fields[num_infos:])

        if mask_fields:
          feat_id = np.asarray(feat_id)
          feat_field = np.asarray(feat_field)
          feat_value = np.asarray(feat_value)
          filter_flag = np.asarray([x not in mask_fields for x in feat_field])
          feat_id = list(feat_id[filter_flag])
          feat_field = list(feat_field[filter_flag])
          feat_value = list(feat_value[filter_flag]) 
        
        if FLAGS.padded_tfrecord:
          feat_id = gezi.pad(feat_id, FLAGS.max_feat_len)
          feat_field = gezi.pad(feat_field, FLAGS.max_feat_len)
          feat_value = gezi.pad(feat_value, FLAGS.max_feat_len)

        feature = {
                    'click': melt.int64_feature(click),
                    'duration': melt.int64_feature(duration),
                    'id': melt.bytes_feature(id),
                    'index': melt.int64_feature(feat_id),
                    'field': melt.int64_feature(feat_field),
                    'value': melt.float_feature(feat_value),
                    'user_click': melt.int64_feature(user_dict['click'][uid]),
                    'user_show': melt.int64_feature(user_dict['show'][uid]),
                    'user_duration': melt.int64_feature(user_dict['duration'][uid])
                   }

        if FLAGS.use_emb:
          feature['cycle_profile_click'] = melt.float_feature(cycle_profile_click)
          feature['cycle_profile_show'] = melt.float_feature(cycle_profile_show)
          feature['cycle_profile_dur'] = melt.float_feature(cycle_profile_dur)
                  
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(record)
        global counter
        with counter.get_lock():
          counter.value += 1
        global flen_counter
        with flen_counter.get_lock():
          flen_counter.value += len(feat_id)


def main(_):
  config.init()
  global dataset
  dataset = Dataset()

  user_dict['show'] = defaultdict(int)
  user_dict['click'] = defaultdict(int)
  user_dict['duration'] = defaultdict(int)
  if os.path.exists('../input/users.txt'):
    for line in open('../input/users.txt'):
      user, show, click, duration = line.rstrip().split('\t')
      user_dict['show'][user] = int(show)
      user_dict['click'][user] = int(click)
      user_dict['duration'][user] = int(duration)

  pool = multiprocessing.Pool()

  files = gezi.list_files(FLAGS.input)
  print('input', FLAGS.input)
  
  if not os.path.exists(FLAGS.out_dir):
    print('make new dir: [%s]' % FLAGS.out_dir, file=sys.stderr)
    os.makedirs(FLAGS.out_dir)

  pool.map(build_features, files)
  pool.close()
  pool.join()

  print('num_records:', counter.value)
  # TODO FIXME flen_counter.value not correct ? for train ?
  print('mean feature len:', flen_counter.value / counter.value)

  out_file = '{}/num_records.txt'.format(FLAGS.out_dir)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  flags.DEFINE_string('input', None, '')
  flags.DEFINE_string('out_dir', None, '')
  flags.DEFINE_bool('use_emb', False, '')
  flags.DEFINE_bool('has_emb', False, '')
  flags.DEFINE_bool('over_write', False, '')
  flags.DEFINE_string('mask_fields', '', '')
  
  absl_app.run(main) 
