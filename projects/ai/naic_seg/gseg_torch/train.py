#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2020-10-11 13:04:10.333525
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
sys.path.append('..')

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.utils.data import DataLoader

import melt as mt
from gseg_torch.model import get_model
from gseg_torch.evaluate import get_eval_fn
from gseg_torch.loss import get_loss_fn
from gseg.dataset import Dataset as TFDataset
from gseg_torch.dataset import *
import gseg


def main(_):
  FLAGS.torch = True
  FLAGS.batch_parse = False
  FLAGS.write_valid = False
  gseg.config.init()
  mt.init()
  model = get_model()

  if not FLAGS.torch_only:
    # 使用tf tfrecord再转换
    mt.fit(model,  
          loss_fn=get_loss_fn(),
          Dataset=TFDataset,
          eval_fn=get_eval_fn()
          ) 
  else:
    # t4 6卡测试 单卡 torch慢 1.4it/s tf 1.8it/s
    # 多卡6卡 tf启动hang住..  
    kwargs = {'num_workers': FLAGS.num_workers, 'pin_memory': True}

    train_ds = Dataset('../input/quarter/train', mode='train')
    train_dl = DataLoader(train_ds, mt.batch_size(), shuffle=True, **kwargs)

    if FLAGS.valid_input or FLAGS.fold is not None:
      valid_ds = Dataset('../input/quarter/train', mode='valid')
      eval_dl = DataLoader(valid_ds, mt.eval_batch_size(), **kwargs)
      valid_dl = DataLoader(valid_ds, mt.batch_size(), **kwargs)

    mt.fit(model,  
          loss_fn=get_loss_fn(),
          dataset=train_dl,
          eval_dataset=eval_dl,
          valid_dataset=valid_dl,
          eval_fn=get_eval_fn()
     )

if __name__ == '__main__':
  app.run(main)  
