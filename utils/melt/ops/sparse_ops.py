#!/usr/bin/env python
# ==============================================================================
#          \file   sparse_ops.py
#        \author   chenghuige  
#          \date   2016-08-16 10:09:41.241790
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
  
# TODO can tf.sparse.to_dense padded_batch   tf.io.VarLenFeature support max_length like tf.io.VarLenFeature(dtype, max_length=100, default_value=0)
def sparse_tensor_to_dense(input_tensor, default_value=0):  
  return tf.sparse.to_dense(input_tensor, default_value=default_value, validate_indices=False)
