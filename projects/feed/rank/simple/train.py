#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import TFRecordDataset
import model as base
import evaluate as ev

import melt

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
  melt.init()
  
  model = getattr(base, FLAGS.model)() 
  
  melt.fit(model,  
           loss_fn=tf.compat.v1.losses.sigmoid_cross_entropy,
           Dataset=TFRecordDataset,
           eval_fn=ev.evaluate,
           valid_write_fn=ev.valid_write,
           write_valid=FLAGS.write_valid)   


if __name__ == '__main__':
  tf.compat.v1.app.run()  
