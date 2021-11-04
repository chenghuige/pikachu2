#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pythonly-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
#   \Description  Notice pythonly train is faster then toch-train using tf dataset but it consume more resource and may fail if run 2 cocurrently
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import multiprocessing

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.nn import functional as F

from pyt.dataset import *
from pyt.model import *
import pyt.model as base
import evaluate as ev

import melt
import gezi
import lele

from projects.feed.rank.src import config
from projects.feed.rank.src import util

from pyt.util import get_optimizer
from pyt.loss import Criterion

import nvidia.dali.tfrecord as tfrec

logging = gezi.logging

def main(_):
  FLAGS.torch_only = True

  #FLAGS.input = '/search/odin/publicData/CloudS/chenghuige/rank_0804_so/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020062400'
  #inputs = [
  #        '/search/odin/publicData/CloudS/chenghuige/rank_0804_so/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020062400',
  #        '/search/odin/publicData/CloudS/chenghuige/rank_0804_so/newmse/data/video_hour_newmse_v1/tfrecords/2020062400',
  #        '/search/odin/publicData/CloudS/chenghuige/rank_0804_so/shida/data/video_hour_shida_v1/tfrecords/2020062400',
  #       ]
  ##inputs = [x.replace('/home/gezi/data/rank', '/search/odin/publicData/CloudS/libowei/rank4') for x in inputs]
  #FLAGS.input = ','.join(inputs)
  ##FLAGS.valid_input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051523'
  #FLAGS.valid_input = '/search/odin/publicData/CloudS/chenghuige/rank_0804_so/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020062402'
  #FLAGS.model_dir = '/tmp/melt'
  #FLAGS.model = 'WideDeep'
  #FLAGS.hash_embedding_type = 'QREmbedding'
  #FLAGS.batch_size = max(FLAGS.batch_size, 512)
  #FLAGS.feature_dict_size = 20000000
  #FLAGS.num_feature_buckets = 3000000
  #FLAGS.use_other_embs = True
  #FLAGS.use_user_emb = True
  #FLAGS.use_doc_emb = True
  #FLAGS.use_history_emb = True
  #FLAGS.fields_pooling = 'dot'

  #FLAGS.use_weight = False
  ##FLAGS.optimizer = 'SparseAdam'
  #FLAGS.optimizer = 'bert-SGD,bert-SGD'
  FLAGS.padded_tfrecord = False
  #FLAGS.is_video = True

  config.init()

  train_files = gezi.list_files(FLAGS.train_input.split('|')[0])
  melt.init()
  fit = melt.fit

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  # loss_fn = nn.BCEWithLogitsLoss()
  loss_fn = Criterion()
  
  features = {}
  example = melt.first_example(train_files[0])

  def to_tfrec(dtype):
    if dtype == np.int64:
      return tfrec.int64, 0
    elif dtype == np.float32:
      return tfrec.float32, 0.
    else:
      return tfrec.int64, 0

  # for key in example:
  #   print(key, example[key].dtype, len(example[key]))
  for key in example:
    dtype, dval = to_tfrec(example[key].dtype)
    if dtype != np.object:
      features[key] = tfrec.FixedLenFeature([len(example[key])], dtype, dval)
    
  # num_workers = 8
  num_workers = 1
  # kwargs = dict(num_workers=num_workers, collate_fn=lele.DictPadCollate())
  kwargs = dict(num_workers=num_workers)
  # kwargs['collate_fn'] = None
  train_ds = TFRecordDataset(train_files, FLAGS.batch_size, features, **kwargs)
  train_dl = train_ds

  i = 0
  model = model.cuda()
  for X, y in tqdm(train_dl, total=train_ds.steps):
    # print(X, y)
    # print(i)
    X, y = lele.to_torch(X, y)
    # y_ = model(X)
    # print(y_) 
    i += 1

  # # exit(0)

  # if FLAGS.valid_input:
  #   # valid_files = gezi.list_files(FLAGS.valid_input)
  #   valid_files = gezi.list_files(FLAGS.valid_input.split('|')[0])
  #   print(FLAGS.valid_input, valid_files)
  #   valid_ds = TFRecordDataset(valid_files, FLAGS.eval_batch_size, features, shuffle=False, **kwargs)
  #   valid_dl = valid_ds

  #   valid_ds2 = TFRecordDataset(valid_files, FLAGS.eval_batch_size, features, shuffle=False, **kwargs)
  #   valid_dl2 = valid_ds2
  
  # eval_fn, eval_keys = util.get_eval_fn_and_keys()
  # valid_write_fn = ev.valid_write

  # weights = None if not FLAGS.use_weight else 'weight'  

  fit(model,  
      loss_fn,
      optimizer=get_optimizer,
      dataset=train_dl,
      # eval_dataset=valid_dl,
      # valid_dataset=valid_dl2,
      # eval_fn=eval_fn,
      # eval_keys=eval_keys,
      # valid_write_fn=valid_write_fn,
      # weights=weights,
     )


if __name__ == '__main__':
  app.run(main)  
