#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2021-06-13 03:29:35.953028
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os

import pandas as pd
import glob
from multiprocessing import Pool, Manager, cpu_count
import pymp

from absl import app 
from wechat.config import *
from wechat.util import *

flags.DEFINE_string('ensemble_pattern', None, '')
flags.DEFINE_alias('epn', 'ensemble_pattern')
flags.DEFINE_list('ensemble_files', [], '')
flags.DEFINE_alias('efs', 'ensemble_files')
flags.DEFINE_list('ensemble_weights', [], '')
flags.DEFINE_alias('ews', 'ensemble_weights')
flags.DEFINE_string('ensemble_version', '', '')
flags.DEFINE_alias('ever', 'ensemble_version')
flags.DEFINE_string('ensemble_dir', 'ensemble', '')
flags.DEFINE_alias('edir', 'ensemble_dir')
flags.DEFINE_bool('write_ensemble_valid', False, '')

def main(_):  
  files = FLAGS.ensemble_files if not FLAGS.ensemble_pattern else glob.glob(FLAGS.ensemble_pattern)
  if not files[0].endswith('.csv'):
    files = [f'../working/online/{FLAGS.ensemble_version}/{x}/submission.csv' for x in files]
  weights = [float(x) for x in FLAGS.ensemble_weights]
  if not weights:
    weights = [1.] * len(files)

  ic(len(files), list(zip(files, weights)))

  models = [x.replace('submission.csv', 'model.h5') for x in files]
  model_exists = [os.path.exists(model) for model in models]
  ic(len(models), list(zip(models, model_exists))) 

#  dfs = Manager().list()
#   ps = min(len(files), cpu_count())
#   with pymp.Parallel(ps) as p:
#     for i in tqdm(p.range(len(files))):
#       df_pred = pd.read_csv(files[i])
#       df_pred = df_pred.sort_values(['userid', 'feedid'])
#       dfs.append(df_pred)
  
  dfs = []
  for i in tqdm(range(len(files))):
    df_pred = pd.read_csv(files[i])
    df_pred = df_pred.sort_values(['userid', 'feedid'])
    dfs.append(df_pred)

  with gezi.Timer('ensemble', print_fn=ic):
    df_pred = ensemble(dfs, weights)

  path_ = os.path.dirname(files[-1])
  mdir = os.path.dirname(path_)
  mname = os.path.basename(path_).split('-')[0]
  odir = f'{mdir}/{FLAGS.ensemble_dir}'
  gezi.try_mkdir(odir)
  with open(f'{odir}/models.txt', 'w') as f:
    for i, (file, weight) in enumerate(zip(files, weights)):
      print(i, file, weight, file=f)
  ofile = f'{odir}/submission.csv'
  ic(ofile)
  df_pred.to_csv(ofile, index=False)

  team_config_file = '/home/tione/notebook/team_config.json'
  if os.path.exists(team_config_file):
    os.system(f'cp {team_config_file} {odir}')


if __name__ == '__main__':
  app.run(main)  
