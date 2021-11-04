#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   checkpoint2savedmodel.py
#        \author   chenghuige  
#          \date   2020-02-07 16:30:23.625384
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import tensorflow as tf
import melt
import gezi

from absl import app, flags
FLAGS = flags.FLAGS
  
def main(_):
  gezi.check_cpu_only()
  model_path = os.path.realpath(sys.argv[1])
  melt.init(tf.Graph())
  sess = melt.get_session()

  predictor = None

  with sess.graph.as_default():
    predictor = melt.Predictor(model_path, sess=sess)   
    assert predictor
    keys = sess.graph.get_all_collection_keys()
    # keys = [key for key in keys if key not in ['cond_context', 'trainable_variables', 'global_step', 'variables']]
    in_keys = [key for key in keys if key.endswith('_feed')]
    out_keys = [key for key in keys if key.startswith('pred')]
    in_ops = [predictor.graph.get_collection(key)[-1] for key in in_keys]
    out_ops = [predictor.graph.get_collection(key)[-1] for key in out_keys]
    inputs_dict = dict(zip(in_keys, in_ops))
    outputs_dict = dict(zip(out_keys, out_ops))
    print(inputs_dict)
    print(outputs_dict)
    path = os.path.realpath(sys.argv[2])
    if os.path.exists(path):
      os.system(f'rm -rf {path}')
    tf.saved_model.simple_save(sess, path, inputs_dict, outputs_dict)

if __name__ == '__main__':
  app.run(main)
  
