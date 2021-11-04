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

import numpy as np
import pandas as pd
import glob

import json
from zipfile import ZIP_DEFLATED, ZipFile
import sklearn.preprocessing as skp
from sklearn.metrics.pairwise import cosine_similarity

import gezi
from gezi import tqdm

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('ensemble_pattern', '*', '')
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
flags.DEFINE_bool('normalize', True, '')
flags.DEFINE_bool('eval', True, '')
flags.DEFINE_bool('calc_sim', True, '')
flags.DEFINE_bool('all', True, '')

from qqbrowser.eval import eval_files

def main(_):  
  files = FLAGS.ensemble_files 
  weights = []
  if not files:
    mark = 'online' if FLAGS.online else 'offline'
    # mark = 'online'
    root = f'../working/{mark}/{FLAGS.ensemble_version}'
    models_file = f'../working/{mark}/{FLAGS.ensemble_version}/models.txt'
    ic(models_file, os.path.exists(models_file))
    if not os.path.exists(models_file) or FLAGS.all:
      if FLAGS.online:
        files = glob.glob(f'{root}/{FLAGS.epn}/pairwise/0/result.json')
      else:
        files = glob.glob(f'{root}/{FLAGS.epn}/pairwise/0/valid.json')
    else:
      for line in open(models_file):
        if line.startswith('#'):
          continue
        model = line.strip().split()[0]
        weight = 1.
        if len(line.strip().split()) == 2:
          weight = float(line.strip().split()[-1])
        weights.append(weight)
        if FLAGS.online:
          files.append(f'{root}/{model}/pairwise/0/result.json')
        else:
          files.append(f'{root}/{model}/pairwise/0/valid.json')
  
  files = [x for x in files if not 'ensemble' in x]
  # files = [x for x in files if not 'wbert' in x]
  # files = [x for x in files if not ('incl_' in x)]
  # files = [x for x in files if ('macbert' in x or 'roberta' in x or 'base' in x) and (not '512' in x)]
  ic(files)

  normalize = FLAGS.normalize

  if not weights:
    weights = [float(x) for x in FLAGS.ensemble_weights]
    if not weights:
      weights = [1.] * len(files)

  ic(files, len(files), list(zip(files, weights)))

  if FLAGS.eval:
    val_files = [x.replace('result.json', 'valid.json').replace('online', 'offline') for x in files]
    ic(val_files, len(val_files))
    assert len(val_files) == len(files)
    for i, file_ in enumerate(val_files):
      ic(i, file_, weights[i], eval_files([file_]))
    nonorm_score = eval_files(val_files, normalize=False, weights=weights, calc_sim=True)
    norm_score = eval_files(val_files, normalize=True, weights=weights)
    ic(nonorm_score, norm_score)
    if norm_score > nonorm_score:
      normalize = True
    else:
      normalize = False

    # normalize = True
    ic(normalize)

  vid_embeddings = []
  for file in tqdm(files):
    with open(file) as f:
      vid_embedding = json.load(f)
      ic(file, len(vid_embedding))
      vid_embeddings.append(vid_embedding)

  m = {}
  for key in vid_embedding:
    m[key] = []

  calc_sim = FLAGS.calc_sim
  m2 = {}
  for i, vid_embedding in tqdm(enumerate(vid_embeddings), total=len(vid_embeddings)):
    try:
      if calc_sim and i > 0:
        sim = 0.
        sims = [0.] * i
      for key in vid_embedding:
        if normalize:
          vid_embedding[key] = skp.normalize([vid_embedding[key]])[0]
        weight = 1. if not weights else weights[i]
        if calc_sim and i > 0:
          # ic(np.mean(np.asarray(m[key]), np.asarray(vid_embedding[key])))
          sim += cosine_similarity([np.mean(np.asarray(m[key]), axis=0)], [vid_embedding[key]])[0][0]
          for j in range(i):
            sims[j] += cosine_similarity([np.asarray(m[key][j])], [vid_embedding[key]])[0][0]
        # assert weight == 1.
        # m[key].append(vid_embedding[key])
        m[key].append(np.asarray([x * weight for x in vid_embedding[key]]))
      if calc_sim and i > 0:
        sim /= len(vid_embedding)
        for j in range(i):
          sims[j] /= len(vid_embedding)
        if i > 0:
          ic(i, files[i], weight, f'{sim:.4f}', [f'{x:.4f}' for x in sims])
    except Exception as e:
      ic(e)
      ic(i, files[i], key, m[key], j, len(m[key]))

  for key in vid_embedding:
    m[key] = list(np.sum(np.asarray(m[key]), axis=0) / sum(weights))

  vid_embedding = m

  out_json_file = 'result.json' 
 
  #model_root/model_dir/pairwise/0/result.json  model_root/model_dir/pairwise/0/
  root = os.path.dirname(files[0])
  # model_root/model_dir/pairwise
  root = os.path.dirname(root)
  # model_root/model_dir
  root = os.path.dirname(root)
  # model_root
  root = os.path.dirname(root)
  root = f'{root}/{FLAGS.ensemble_dir}'
  ic(root)
  # exit(0)
  gezi.try_mkdir(root)
  out_json = f'{root}/{out_json_file}'
  ic(out_json)
  with open(out_json, 'w') as f:
    json.dump(vid_embedding, f)

  with ZipFile(f'{root}/result.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
    zip_file.write(out_json, out_json_file)

if __name__ == '__main__':
  app.run(main)  
