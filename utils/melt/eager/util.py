#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2018-09-03 12:05:03.098236
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

import gezi
import melt 
logging = gezi.logging

import inspect

# def grad(model, x, y, loss_fn, weights=1.0, hvd=None):
#   with tf.GradientTape() as tape:
#     if 'training' in inspect.getargspec(model.call).args:
#       y_ = model(x, training=True)
#     else:
#       y_ = model(x)
#     if 'weights' in inspect.getargspec(loss_fn).args:
#       if isinstance(weights, str):
#         loss = loss_fn(y, y_, weights=x[weights])
#       else: 
#         loss = loss_fn(y, y_, weights=weights)
#     else:
#       loss = loss_fn(y, y_)
#     if hvd is not None:
#       tape = hvd.DistributedGradientTape(tape)
#   return loss, tape.gradient(loss, model.trainable_variables)

def grad(model, x, y, loss_fn, hvd=None):
  with tf.GradientTape() as tape:
    if loss_fn is not None:
      loss = loss_fn(x, y)
    else:
      _ = model(x, training=True)
      loss = 0.
    loss_ = loss
    if model.losses:
      ic(model.losses)
      loss += sum(model.losses)
    # if hvd is not None:
    #   tape = hvd.DistributedGradientTape(tape, sparse_as_dense=FLAGS.sparse_to_dense)
  gradients = tape.gradient(loss, model.trainable_variables)
  return loss_, gradients


def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)

def restore(model, ckpt_dir=None):
  if not ckpt_dir:
    ckpt_dir = FLAGS.model_dir + '/ckpt'
  
  if os.path.exists(ckpt_dir + '.index'):
    latest_checkpoint = ckpt_dir
  else:
    latest_checkpoint = melt.latest_checkpoint(ckpt_dir)

  logging.info('Latest checkpoint:', latest_checkpoint)

  checkpoint = tf.train.Checkpoint(model=model)      
  
  # TODO check return value, verify if it is restore ok ?
  checkpoint.restore(latest_checkpoint)
