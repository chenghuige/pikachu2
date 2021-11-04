#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2020-04-12 20:31:52.757244
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import flags
FLAGS = flags.FLAGS

from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils import shuffle

import gezi
logging = gezi.logging

eval_langs = ['es', 'it', 'tr']

def evaluate(y_true, y_pred, x):
  if FLAGS.task == 'toxic':
    try:
      y_true = y_true[:,0]
      y_pred = y_pred[:,0]
    except Exception:
      pass
    if y_pred.max() > 1. or y_pred.min() < 0:
      y_pred = gezi.sigmoid(y_pred)
    result = OrderedDict()
    loss = log_loss(y_true, y_pred)
    result['loss'] = loss
    
    auc = roc_auc_score(y_true, y_pred)
    result['auc/all'] = auc
      
    if 'lang' in x:
      x['y_true'] = y_true
      x['pred'] = y_pred
      x['lang'] = gezi.decode(x['lang'])

      df = pd.DataFrame(x) 
      df = shuffle(df)
      gezi.pprint(pd.concat([df[df.y_true==0].sample(5), df[df.y_true==1].sample(5)]), 
                  print_fn=logging.info, desc='preds:', format='%.4f')

      for lang in eval_langs:
        df_ = df[df.lang==lang]
        if len(df_):
          auc = roc_auc_score(df_.y_true, df_.pred)
          result[f'auc/{lang}'] = auc
  elif FLAGS.task == 'lang':
    results = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    result = {'acc': np.sum(results) / len(results)}

  return result

if __name__ == '__main__':
  import glob
  # vdf = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
  # vdf = vdf.sort_values('id')
  input_ = sys.argv[1] 
  if input_.endswith('.csv'):
    ifile = input_
    df = pd.read_csv(ifile)
  else:
    dfs = [pd.read_csv(file_) for file_ in glob.glob(f'{input_}/valid_*.csv')]
    df = pd.concat(dfs)
  df = df.sort_values('id')      
      
  y_true = df.label.values
  y_pred = df.pred.values
  # x = {}
  # x['lang'] = [vdf[vdf.id==id].id for id in df.id]
  # print(len(df))
  # print(y_true)
  # print(y_pred)
  # gezi.pprint(evaluate(y_true, y_pred, x))
  # print(np.mean([roc_auc_score(dfs[i].label.values, dfs[i].pred.values) for i in range(len(dfs))]))
  print('full-auc:', roc_auc_score(y_true, y_pred))
  try:
    aucs = [roc_auc_score(df.label.values, df.pred.values) for df in dfs]
    auc = np.mean(aucs)
    print('mean-auc:', auc)
  except Exception:
    pass


  
