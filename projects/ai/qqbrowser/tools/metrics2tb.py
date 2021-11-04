#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics2tb.py
#        \author   chenghuige  
#          \date   2021-10-16 12:42:22.176337
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob
import pandas as pd
import gezi

ver = sys.argv[1]  
root = f'../working/offline/{ver}'

models = glob.glob(f'{root}/*/pairwise')
ic(models)
log_root = f'{root}/metrics'

gezi.remove_dir(log_root)
gezi.try_mkdir(log_root)
ic(log_root)

for model in models:
  model_name = os.path.basename(os.path.dirname(model))
  model_name = model_name.replace('model.', '')
  if 'incl' in model_name:
    continue
  log_dir = f'{log_root}/{model_name}/pairwise'
  ic(log_dir)
  sw = gezi.SummaryWriter(log_dir)
  files = glob.glob(f'{model}/*/metrics.csv')
  dfs = [pd.read_csv(file) for file in files]
  ic(len(dfs))
  df = pd.concat(dfs)
  if not len(df):
    continue
  step = df.step.max()
  df = df[df.step == step]
  ic(len(df))
  if len(df) != 5:
    continue
  df = df.sort_values(['ntime'])
  ic(df.spearmanr)
  ic(df.spearmanr.mean())
  for i in range(len(df)):
    sw.scalar('spearmanr/fold', df.spearmanr.values[i], step=i)
    sw.scalar('spearmanr/mean', df.spearmanr.values[:i + 1].mean(), step=i)

  log_dir = log_dir.replace('pairwise', 'pointwise')
  ic(log_dir)
  sw = gezi.SummaryWriter(log_dir)
  model = model.replace('pairwise', 'pointwise')
  df = pd.read_csv(f'{model}/metrics.csv')
  metrics = [x for x in df.columns if x not in ['step', 'ntime', 'epoch', 'insts']]
  ic(metrics)
  for metric in metrics:
    metric_ = metric.replace('pointwise/', '')
    for i in range(len(df)):
      sw.scalar(metric_, df[metric].values[i], step=i)
    


