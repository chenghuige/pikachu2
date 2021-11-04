#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:02.802049
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from genericpath import exists

import sys

from sklearn.utils.validation import assert_all_finite
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
import glob
import pandas as pd
import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt as mt

from qqbrowser.models_factory import get_model
from qqbrowser.dataset import Dataset
import qqbrowser.eval as ev
from qqbrowser import config
from qqbrowser.config import *
from qqbrowser import util
from qqbrowser.loss import get_loss

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  config.init()
  mt.init()

  model_dir = FLAGS.model_dir
  if util.is_pairwise():
    pretrain = f'{FLAGS.model_dir}/pointwise'
    pretrain = pretrain.replace('online', 'offline')
    if os.path.exists(pretrain):
      FLAGS.pretrain = pretrain
    elif FLAGS.pretrain:
      # assert not os.path.exists( f'{FLAGS.model_dir}/pointwise')
      root = os.path.dirname(FLAGS.model_dir)
      FLAGS.pretrain = f'{root}/{FLAGS.pretrain}'
    FLAGS.model_dir = f'{FLAGS.model_dir}/pairwise/{FLAGS.fold_}'
    FLAGS.model_name += f'/pairwise/{FLAGS.fold_}'
  else:
    FLAGS.model_dir = f'{FLAGS.model_dir}/pointwise'
    FLAGS.model_name += '/pointwise'

  ic(model_dir, FLAGS.model_dir, FLAGS.model_name, FLAGS.pretrain)
  if FLAGS.online:
    assert FLAGS.fold_ == 0
  if FLAGS.work_mode != 'train' and FLAGS.online:
    if not os.path.exists(f'{FLAGS.model_dir}/model.h5'):
      model_dir_ = os.path.dirname(FLAGS.model_dir)
      gezi.try_mkdir(FLAGS.model_dir)
      ic('copy model', f'{model_dir_}/model.h5', f'{FLAGS.model_dir}/model.h5')
      gezi.copyfile(f'{model_dir_}/model.h5', f'{FLAGS.model_dir}/model.h5')

  gezi.try_mkdir(FLAGS.model_dir)
  
  # if os.path.exists(f'{FLAGS.model_dir}/done.txt'):
  if util.is_pairwise():
    if os.path.exists(f'{FLAGS.model_dir}/model.h5') and FLAGS.work_mode == 'train':
      ic(FLAGS.model_dir, 'exists')
      exit(0)
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = get_model(FLAGS.model)
    model.eval_keys = ['vid', 'vid1', 'vid2', 'relevance', 'pos']
    model.str_keys = ['vid', 'vid1', 'vid2']
    model.out_keys = ['final_embedding', 'top_tags', 'top_weights', 'top_tags1', 'top_weights1', 
                      'top_tags2', 'top_weights2', 'final_embedding1', 'final_embedding2']
    
    if not FLAGS.baseline_dataset:
      if not FLAGS.lm_target:
        # 走这里
        fit(model,  
            loss_fn=get_loss(model),
            Dataset=Dataset,
            eval_fn=ev.evaluate,
            valid_write_fn=ev.valid_write,
            infer_write_fn=ev.infer_write,
            ) 
      else:
        fit(model,  
            loss_fn=get_loss(model),
            Dataset=Dataset,
          ) 
    else:
      train_dataset, (eval_dataset, val_dataset), test_dataset = util.get_datasets()
      fit(model,  
          loss_fn=get_loss(model),
          dataset=train_dataset,
          eval_dataset=eval_dataset, 
          valid_dataset=val_dataset,
          test_dataset=test_dataset,
          eval_fn=ev.evaluate,
          valid_write_fn=ev.valid_write,
          infer_write_fn=ev.infer_write,
          ) 

  if util.is_pairwise() and (not FLAGS.online):
    log_dir = f'{model_dir}/final'
    sw = gezi.SummaryWriter(log_dir)
    files = glob.glob(f'{model_dir}/pairwise/*/metrics.csv')
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs)
    df_ = df.groupby(['step'])['spearmanr'].mean().reset_index()
    df_.to_csv(f'{model_dir}/spearmanr.csv', index=False)
    step = df.step.max()
    df = df[df.step == step]
    df = df.sort_values(['ntime'])
    ic(df.spearmanr)
    ic(df.spearmanr.mean())
    sw.scalar('spearmanr/fold', df.spearmanr.values[-1], step=FLAGS.fold_)
    sw.scalar('spearmanr/mean', df.spearmanr.mean(), step=FLAGS.fold_)

if __name__ == '__main__':
  app.run(main)  
