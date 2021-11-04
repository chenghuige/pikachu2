#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
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

from tensorflow import keras

import gezi
logging = gezi.logging
import melt

import model as base
from dataset import Dataset
from config import *
from evaluate import *
from util import *

def main(_):
  FLAGS.eager = True
  melt.init()
  model = getattr(base, FLAGS.model)() 
  model_dir = FLAGS.model_dir
  melt.eager.restore(model, model_dir)
  dataset = Dataset('valid')
  files = gezi.list_files('../input/tfrecords/test')
  total = melt.get_num_records(files) 
  batch_size = 512
  batches = dataset.make_batch(batch_size=batch_size, filenames=files, repeat=False)
  num_steps = -int(-total // batch_size)
  with open(f'{model_dir}/submission.csv', 'w') as out:
    print('user_id,predicted_age,predicted_gender', file=out)
    for i, (x, _) in tqdm(enumerate(batches), total=num_steps, ascii=True, desc='loop'):
      model(x)
      uids = x['id'].numpy()
      ages = model.pred_age.numpy()
      genders = model.pred_gender.numpy()

      ages = np.asarray([to_age(x) for x in ages])
      genders = np.asarray([to_gender(x) for x in genders])

      for user_id, age, gender in zip(uids, ages, genders):
        print(user_id, age, gender, sep=',', file=out)

if __name__ == '__main__':
  app.run(main)  
