#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-26 23:00:24.215922
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

import melt 
logging = melt.logging
import numpy as np

from projects.feed.rank.src.config import *

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import nvidia.dali.tfrecord as tfrec

class CommonPipeline(Pipeline):
  def __init__(self, batch_size, num_threads, device_id):
    super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)
    pass

  def base_define_graph(self, inputs, labels):
    pass

class TFRecordPipeline(CommonPipeline):
  def __init__(self, batch_size, features, path, index_path=None, shuffle=False, num_threads=64, device_id=0, num_gpus=1):
    super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)
    if index_path is None:
      index_path = [x + '.idx' for x in path]

    # features = {
    #             #'click': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #             'duration': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #             'index': tfrec.FixedLenFeature([138], tfrec.int64, 0),
    #             'field': tfrec.FixedLenFeature([138], tfrec.int64, 0),
    #             'value': tfrec.FixedLenFeature([138], tfrec.float32, 0.),
    #             # 'id': tfrec.FixedLenFeature((), tfrec.string, ''),
    #             'uid': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #             'did': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #             # 'time_interval': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #             # 'time_weekday': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #             # 'timespan_interval': tfrec.FixedLenFeature((), tfrec.int64, 0),
    #            }

    self.keys = list(features.keys())
    
    self.input = ops.TFRecordReader(path=path,
                                    index_path=index_path,
                                    features=features,
                                    random_shuffle=shuffle,
                                    num_shards=num_gpus,
                                    shard_id=device_id,
                                    )

  def define_graph(self):
    inputs = self.input(name="Reader")
    return tuple(inputs.values())

