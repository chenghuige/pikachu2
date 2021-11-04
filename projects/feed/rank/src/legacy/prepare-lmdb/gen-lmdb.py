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

import gezi
import melt
from text_dataset import Dataset

import tensorflow as tf

import numpy as np

import pyxis as px

dataset = None
user_dict = {}

total = 568098

def get_out_dir(infile):
  infile_ = os.path.basename(infile)
  odir_ = infile_ 
  odir = os.path.join(FLAGS.out_dir, odir_)
  return odir

def build_features(infile):
  odir = get_out_dir(infile)
  if not FLAGS.over_write:
    if os.path.exists(odir):
      print('%s exists' % odir)
      return
  print('----------writing to', odir)
  with px.Writer(dirpath=odir, map_size_limit=100000, ram_gb_limit=10) as writer:
    for line in tqdm(open(infile), total=total, ascii=True):
      fields = line.rstrip().split('\t')
      if len(fields) > 4:
        click = int(fields[0])
        duration = int(fields[1])
        if duration > 60 * 60 * 12:
          duration = 60
        id = '{}\t{}'.format(fields[2], fields[3])
        uid = fields[2]
        feat_id, feat_field, feat_value, cycle_profile_click, cycle_profile_show, cycle_profile_dur = dataset.get_feat_portrait(fields[4:])

        feature = {
                    'click': np.asarray([click]),
                    'duration': np.asarray([duration]),
                    'id': np.asarray([id]),  
                    'index': np.asarray([feat_id]),
                    'field': np.asarray([feat_field]),
                    'value': np.asarray([feat_value], dtype=np.float32),
                    'user_click': np.asarray([user_dict['click'][uid]], dtype=np.float32),
                    'user_show': np.asarray([user_dict['show'][uid]], dtype=np.float32),
                    'user_duration': np.asarray([user_dict['duration'][uid]], dtype=np.float32)
                  }
        
        if FLAGS.use_emb:
          feature['cycle_profile_click'] = np.asarray([cycle_profile_click])
          feature['cycle_profile_show'] = np.asarray([cycle_profile_show])
          feature['cycle_profile_dur'] = np.asarray([cycle_profile_dur])

        writer.put_samples(feature)
        global counter
        with counter.get_lock():
          counter.value += 1


def main(_):
  global dataset
  dataset = Dataset()

  user_dict['show'] = {}
  user_dict['click'] = {}
  user_dict['duration'] = {}
  for line in open('../input/users.txt'):
    user, show, click, duration = line.rstrip().split('\t')
    user_dict['show'][user] = int(show)
    user_dict['click'][user] = int(click)
    user_dict['duration'][user] = int(duration)
  
  pool = multiprocessing.Pool()

  files = glob.glob(FLAGS.input)
  print('input', FLAGS.input)
  

  if not os.path.exists(FLAGS.out_dir):
    print('make new dir: [%s]' % FLAGS.out_dir, file=sys.stderr)
    os.makedirs(FLAGS.out_dir)

  pool.map(build_features, files)
  pool.close()
  pool.join()

  print('num_records:', counter.value)

  out_file = '{}/num_records.txt'.format(FLAGS.out_dir)
  gezi.write_to_txt(counter.value, out_file)

if __name__ == '__main__':
  flags.DEFINE_string('input', None, '')
  flags.DEFINE_string('out_dir', None, '')
  flags.DEFINE_string('mode', None, '')
  flags.DEFINE_bool('use_emb', True, '')
  flags.DEFINE_bool('over_write', False, '')

  absl_app.run(main) 
