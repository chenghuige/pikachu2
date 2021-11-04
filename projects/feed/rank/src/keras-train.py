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

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from dataset import *
from model import *
import model as base
import evaluate as ev
import loss

import tensorflow as tf
from absl import flags, app
FLAGS = flags.FLAGS

import gezi
logging = gezi.logging

import husky

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K


def main(_):
  melt.init()
  fit = melt.fit

  loss.init()

  Dataset = TFRecordDataset
  FLAGS.hash_embedding_type = 'SimpleEmbedding'

  model_name = 'Model'
  model = getattr(base, model_name)() 

  loss_fn = loss.get_loss_fn()

  logging.info('loss_fn', loss_fn)

  def get_variables_list(all_vars):
    variables_list = None
    if FLAGS.optimizers:
      assert len(FLAGS.optimizers.split(',')) == 2, 'only support 2 optimizers now ' + FLAGS.optimizers
      all_vars = tf.compat.v1.trainable_variables()
      wide_wars = [x for x in all_vars if x.name.startswith('wide_deep/wide')]
      gezi.sprint(wide_wars)
      deep_wars = [x for x in all_vars if not x.name.startswith('wide_deep/wide')]
      gezi.sprint(deep_wars)
      assert wide_wars
      assert deep_wars
      variables_list = [deep_wars, wide_wars]
      gezi.sprint(variables_list)
      return variables_list
    else:
      return None

  #eval_fn = ev.evaluate if not 'METRIC' in os.environ else ev.evaluate2
  eval_fn = ev.evaluate 
  eval_keys = None
  #if FLAGS.eval_rank or 'EVAL_RANK' in os.environ:
  if FLAGS.eval_rank and not gezi.env_val('EVAL_RANK') == '0':
    def eval_fn(y, y_, x, other):
      return ev.evaluate_rank(y, y_, x, other, group=FLAGS.eval_group)
    eval_keys = ['id', 'duration']

  weights = None if not FLAGS.duration_weight else 'weight'

  valid_write_fn = ev.valid_write_duration
  out_hook = ev.out_hook

  dataset, eval_dataset, valid_dataset = melt.apps.get_datasets(Dataset)
  print('----dataset', dataset)

  model.compile(
    loss=loss_fn,
    optimizer=Adam(lr=1e-3, epsilon=1e-6))

  steps_per_epoch=-(-FLAGS.num_train // FLAGS.batch_size)
  validation_steps=-(-FLAGS.num_valid // FLAGS.batch_size)

  callback = husky.callbacks.EvalCallback(eval_dataset, eval_fn)
  callback.set_model(model)
  callback.steps_per_epoch = steps_per_epoch
  callback.steps = validation_steps
  callback.num_valid_examples = FLAGS.num_valid
  callback.out_hook = out_hook
  callback.eval_keys=eval_keys
  SummaryWriter = gezi.SummaryWriter if not tf.executing_eagerly() else gezi.EagerSummaryWriter
  callback.logger = SummaryWriter(FLAGS.model_dir)  
  callbacks = [callback, husky.callbacks.ValidLoss(valid_dataset)]

  model.fit(dataset,
            epochs=FLAGS.num_epochs,
            steps_per_epoch=steps_per_epoch,
            # batch_size=FLAGS.batch_size,
            # validation_data=eval_dataset,
            # validation_steps=1,
            callbacks=callbacks)

if __name__ == '__main__':
  app.run(main)  
