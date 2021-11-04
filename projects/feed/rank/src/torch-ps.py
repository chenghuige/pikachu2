#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
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

import torch
from torch import nn
from torch.nn import functional as F

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

import argparse
# import deepspeed
from tqdm import tqdm
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

import threading
import queue
import time

class AsyncWorker(threading.Thread):
    def __init__(self, dataloader, total):
        threading.Thread.__init__(self)
        self.req_queue = queue.Queue()
        self.ret_queue = queue.Queue()
        self.dataloader = iter(dataloader)
        self.total = total
        self.prefetch_idx = 3
        for i in range(self.prefetch_idx):
            self.req_queue.put(1)

    def run(self):
        while True:
            dataset_type = self.req_queue.get(block=True)
            if dataset_type is None:
                break
            batch = next(self.dataloader)
            self.req_queue.task_done()
            self.ret_queue.put(batch)

    def get(self):
        batch = self.ret_queue.get()
        self.ret_queue.task_done()
        return batch

    def __iter__(self):
      return self.get()

    def prefetch(self):
        if self.prefetch_idx < self.total:
            self.req_queue.put(1)
            self.prefetch_idx += 1

    def stop(self):
        self.req_queue.put(None)

def main(_):
  argv = open('./flags/torch/dlrm/command.txt').readline().strip().replace('data_version=2', 'data_version=1').split()
  FLAGS(argv)

  FLAGS.torch = True
  FLAGS.torch_only = False
  FLAGS.loop_train = False

  FLAGS.input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520'
  # inputs = [
  #         '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520',
  #         '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051521',
  #         '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051522',
  #        ]
  # inputs = [x.replace('/home/gezi/data/rank', '/search/odin/publicData/CloudS/libowei/rank4') for x in inputs]
  inputs = [
          '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051520',
          '/search/odin/publicData/CloudS/libowei/rank4/newmse/data/video_hour_newmse_v1/tfrecords/2020051520',
          '/search/odin/publicData/CloudS/libowei/rank4/shida/data/video_hour_shida_v1/tfrecords/2020051520',
         ]
  FLAGS.input = ','.join(inputs)
  FLAGS.valid_input = '/search/odin/publicData/CloudS/libowei/rank4/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051522'
  #FLAGS.valid_input = '/home/gezi/data/rank/sgsapp/data/video_hour_sgsapp_v1/tfrecords/2020051523'

  FLAGS.model_dir = '/tmp/melt'
  os.system('rm -rf %s' % FLAGS.model_dir)
  # FLAGS.model = 'WideDeep'
  # FLAGS.hash_embedding_type = 'QREmbedding'
  FLAGS.batch_size = max(FLAGS.batch_size, 512)
  # FLAGS.feature_dict_size = 20000000
  # FLAGS.num_feature_buckets = 3000000
  # FLAGS.use_user_emb = True
  # FLAGS.use_doc_emb = True
  # FLAGS.use_history_emb = True
  # FLAGS.fields_pooling = 'dot'
  FLAGS.min_free_gpu_mem = 0
  FLAGS.max_used_gpu_mem = 20000000000
  # FLAGS.sparse_emb = False

  config.init()
  melt.init()
  fit = melt.fit
 
  Dataset = util.prepare_dataset()

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  loss_fn = Criterion()
  
  eval_fn, eval_keys = util.get_eval_fn_and_keys()
  valid_write_fn = ev.valid_write
  out_hook = ev.out_hook

  weights = None if not FLAGS.use_weight else 'weight'

  fit(model,  
      loss_fn,
      Dataset,
      eval_fn=eval_fn,
      eval_keys=eval_keys,
      valid_write_fn=valid_write_fn,
      out_hook=out_hook,
      weights=weights)

  print('-----------DONE')
   
if __name__ == '__main__':
  main(None)
  
