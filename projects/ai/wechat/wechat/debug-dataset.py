#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-tfrecord.py
#        \author   chenghuige  
#          \date   2019-09-11 11:00:01.818073
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
import melt as mt
import gezi
from gezi import tqdm

def main(_):
  record_files = gezi.list_files(sys.argv[1])
  print(record_files)
  t = tqdm(record_files)
  for record_file in t:
    t.set_postfix({'file': record_file})
    for i, item in enumerate(tf.data.TFRecordDataset(record_file)):
      x = mt.decode_example(item)
  #     print(x.keys())
  #     print(len(x['negs_spans']))
      # if len(x['negs_spans']) != 50:
      #   print(x)
      #   print(x['negs_spans'])
      #   print(len(x['negs_spans']), len(x['negs']), len(x['poss']))
      #   print(x['userid'], x['date'])
      #   print(x['negs'])
      #   print(list(zip(range(len(x['negs'])), x['negs'])))
      #   break
      if x['userid'][0] == 196786:
        print(x)
      # print(x['userid'], len(x['poss']))
      # if len(x['poss']) == 1:
      #   print(x['userid'], x['poss'])

if __name__ == '__main__':
  app.run(main)  
  
