#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tfrecord2txt.py
#        \author   chenghuige  
#          \date   2019-09-16 11:27:57.605337
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os
from tfrecord_lite import decode_example

from tqdm import tqdm

import melt

def main(argv):
  total = melt.get_num_records([argv[1]])
  with open(sys.argv[2], 'w') as out:
    for it in tqdm(tf.compat.v1.python_io.tf_record_iterator(argv[1]), total=total):
      x = decode_example(it)
      # print(x['id'][0].decode(), x['duration'][0], len(x['index']), ','.join(map(str, x['index'])), ','.join(map(str, x['value'])), ','.join(map(str, x['field'])), sep='\t', file=out)
      print(x['id'][0].decode(), x['duration'][0], len(x['index']), ','.join('{}:{}:{}'.format(x, y, z) for x, y, z in zip(x['index'], x['field'], x['value'])), sep='\t', file=out)


if __name__ == '__main__':
  app.run(main)  
  
