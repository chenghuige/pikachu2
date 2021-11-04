#!/usr/bin/env python
# ==============================================================================
#          \file   util.py
#        \author   chenghuige
#          \date   2016-08-16 19:32:41.443712
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.core.numeric import False_

import tensorflow as tf
from absl import flags

from melt.ops.ops import embedding_lookup_mean

FLAGS = flags.FLAGS
from tensorflow.python import pywrap_tensorflow

# import tensorflow.contrib.slim as slim

import sys
import os
import glob
import traceback
import inspect
import six
# import subprocess
import numpy as np
import re
import copy
# import pandas as pd
from datetime import datetime
import collections
from collections import Iterable, OrderedDict
import fcntl
import gezi
# from tqdm.notebook import tqdm
from gezi import tqdm
import melt
try:
  import wandb
  HAS_WANDB = True
except Exception:
  HAS_WANDB = False

logging = gezi.logging

# from tensorflow.compat.v1 import keras
# from tensorflow.compat.v1.keras import backend as K

from tensorflow import keras
from tensorflow.keras import backend as K

from gezi.summary import SummaryWriter

from tensorflow.python.platform import gfile

from husky.callbacks.tqdm_progress_bar import TQDMProgressBar


# TODO FIXME should use below but now not work
def create_restore_fn(checkpoint, model_name, restore_model_name):
  model_name = gezi.pascal2gnu(model_name)
  restore_model_name = gezi.pascal2gnu(restore_model_name)

  variables_to_restore = tf.compat.v1.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name)
  assert variables_to_restore, tf.compat.v1.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES)

  prefix = '%s/%s' % (model_name, restore_model_name)

  # remove model_name
  def name_in_checkpoint(var):
    return var.op.name.replace(prefix, restore_model_name)

  variables_to_restore = {
      name_in_checkpoint(var): var
      for var in variables_to_restore
      if var.op.name.startswith(prefix)
  }

  varnames_in_checkpoint = melt.get_checkpoint_varnames(checkpoint)
  # FIXME wrong..
  variables_to_restore = {var2: var for var2 in varnames_in_checkpoint}

  saver = tf.train.Saver(variables_to_restore)

  def restore_fn(sess):
    timer = gezi.Timer('restore var from %s %s' %
                       (restore_model_name, checkpoint))
    saver.restore(sess, checkpoint)
    timer.print()

  return restore_fn


def to_functional_model(model, Dataset):
  dataset = Dataset('train')
  dataset.make_batch()
  if not dataset.has_varlen_feats:
    inputs = dataset.get_inputs()
    model = model.get_model(inputs)
  else:
    logging.warning('Has varlen feats, do not convert to functional model')
  return model


def out_hook(model, keys=[]):
  m = {}
  for key in keys:
    m[key] = getattr(model, key)
  return m


# https://github.com/tensorflow/tensorflow/blob/a7b34f656e78f126aad08a0ef4b29d71469a81cd/tensorflow/python/keras/engine/training.py#L2723
if tf.__version__ > '2':
  from tensorflow.python.keras.engine import data_adapter
  from tensorflow.python.util import nest
  from tensorflow.python.keras.utils import tf_utils

  from tensorflow.python.framework import sparse_tensor
  from tensorflow.python.ops.ragged import ragged_tensor
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import sparse_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops.ragged import ragged_concat_ops
  from tensorflow.python.keras import callbacks as callbacks_module

  def concat(tensors, axis=0):
    """Concats `tensor`s along `axis`."""
    if isinstance(tensors[0], sparse_tensor.SparseTensor):
      return sparse_ops.sparse_concat_v2(axis=axis, sp_inputs=tensors)
    if isinstance(tensors[0], ragged_tensor.RaggedTensor):
      return ragged_concat_ops.concat(tensors, axis=axis)
    return array_ops.concat(tensors, axis=axis)

  # https://github.com/keras-team/keras/blob/master/keras/engine/training.py
  def _minimum_control_deps(outputs):
    """Returns the minimum control dependencies to ensure step succeeded."""
    if tf.executing_eagerly():
      return []  # Control dependencies not needed.
    outputs = tf.nest.flatten(outputs, expand_composites=True)
    for out in outputs:
      # Variables can't be control dependencies.
      if not isinstance(out, tf.Variable):
        return [out]  # Return first Tensor or Op from outputs.
    return []  # No viable Tensor or Op to use for control deps.

  # TODO customed evaluate to return loss and also X infos
  # TODO kaggle now tf 2.2 will not show progress bar as not use callback
  # TODO FIXME mind tpu tf 2.4 will hang if drop_remainder is False when evaluating, training no problem why
  def predict(model,
              x,
              steps=None,
              dump_inputs=True,
              callbacks=None,
              write_fn=None,
              desc='predict',
              verbose=0):
    outputs = None
    with model.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.DataHandler(x=x,
                                              batch_size=None,
                                              steps_per_epoch=steps,
                                              initial_epoch=0,
                                              epochs=1)
      steps = data_handler.inferred_steps

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        if verbose:
          logging.info(desc)
        callbacks = callbacks_module.CallbackList(callbacks,
                                                  add_history=True,
                                                  add_progbar=verbose != 0,
                                                  model=model,
                                                  verbose=verbose,
                                                  epochs=1,
                                                  steps=steps)

      if tf.__version__ < '2.3':
        callbacks._t_enter_batch = 0

      model.predict_function_outhook = model.make_predict_outhook_function(
          dump_inputs)
      model._predict_counter.assign(0)
      batch_outputs = None
      callbacks.on_predict_begin()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        with data_handler.catch_stop_iteration():
          ## TODO 在colab tpu显示不正确 本地gpu跑到是ok
          # for step in tqdm(data_handler.steps(), total=steps, ascii=True, desc=desc):
          for step in data_handler.steps():
            callbacks.on_predict_batch_begin(step)
            tmp_batch_outputs = model.predict_function_outhook(iterator)
            #               print('--------------', tmp_batch_outputs)
            if write_fn is not None:
              write_fn(tmp_batch_outputs)
            if not gezi.get('tpu'):
              with tf.device('/cpu:0'):
                if not isinstance(tmp_batch_outputs, dict):
                  batch_outputs = tf.identity(
                      tmp_batch_outputs)  # No error, now safe to assign.
                else:
                  batch_outputs = {}
                  # ic(list(tmp_batch_outputs.keys()))
                  for key in tmp_batch_outputs:
                    # ic(key, tmp_batch_outputs[key])
                    batch_outputs[key] = tf.identity(tmp_batch_outputs[key])
            else:
              # batch_outputs = tmp_batch_outputs
              if not isinstance(tmp_batch_outputs, dict):
                batch_outputs = tmp_batch_outputs
              else:
                batch_outputs = {}
                for key in tmp_batch_outputs:
                  batch_outputs[key] = tmp_batch_outputs[key]

            if outputs is None:
              outputs = nest.map_structure(lambda batch_output: [batch_output],
                                           batch_outputs)
            else:
              if write_fn is None:
                nest.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs, batch_outputs)
            if tf.__version__ >= '2.3':
              end_step = step + data_handler.step_increment
            else:
              end_step = step
            # if not gezi.get('tpu'):
            callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
      if batch_outputs is None:
        raise ValueError('Expect x to be a non-empty array or dataset.')
      callbacks.on_predict_end()

    with tf.device('/cpu:0'):
      all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
      if tf.__version__ < '2.5':
        return tf_utils.to_numpy_or_python_type(all_outputs)
      else:
        return tf_utils.sync_to_numpy_or_python_type(all_outputs)

  # 很奇怪 之前 貌似 loop是能够tpu 处理 带string key的情况 但是目前tf2.4似乎不可以了.. TODO
  def loop(model, x, steps=None, callbacks=None, desc='loop', verbose=0):
    outputs = None
    with model.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.DataHandler(x=x,
                                              batch_size=None,
                                              steps_per_epoch=steps,
                                              initial_epoch=0,
                                              epochs=1)
      steps = data_handler.inferred_steps

    # Container that configures and calls `tf.keras.Callback`s.
    if not isinstance(callbacks, callbacks_module.CallbackList):
      if verbose:
        logging.info(desc)
      callbacks = callbacks_module.CallbackList(callbacks,
                                                add_history=True,
                                                add_progbar=verbose != 0,
                                                model=model,
                                                verbose=verbose,
                                                epochs=1,
                                                steps=steps)

      if tf.__version__ < '2.3':
        callbacks._t_enter_batch = 0

      model.loop_function = model.make_loop_function()
      model._predict_counter.assign(0)
      batch_outputs = None
      callbacks.on_predict_begin()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        with data_handler.catch_stop_iteration():
          # for step in tqdm(data_handler.steps(), total=steps, ascii=True, desc=desc):
          for step in data_handler.steps():
            callbacks.on_predict_batch_begin(step)
            tmp_batch_outputs = model.loop_function(iterator)
            if tf.__version__ >= '2.3':
              if data_handler.should_sync:
                context.async_wait()
            else:
              if not data_handler.inferred_steps:
                context.async_wait()
            # batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
            if not gezi.get('tpu'):
              with tf.device('/cpu:0'):
                if not isinstance(tmp_batch_outputs, dict):
                  batch_outputs = tf.identity(
                      tmp_batch_outputs)  # No error, now safe to assign.
                else:
                  batch_outputs = {}
                  for key in tmp_batch_outputs:
                    batch_outputs[key] = tf.identity(tmp_batch_outputs[key])
            else:
              # batch_outputs = tmp_batch_outputs
              if not isinstance(tmp_batch_outputs, dict):
                batch_outputs = tmp_batch_outputs
              else:
                batch_outputs = {}
                for key in tmp_batch_outputs:
                  batch_outputs[key] = tmp_batch_outputs[key]

            if outputs is None:
              outputs = nest.map_structure(lambda batch_output: [batch_output],
                                           batch_outputs)
            else:
              nest.map_structure_up_to(
                  batch_outputs,
                  lambda output, batch_output: output.append(batch_output),
                  outputs, batch_outputs)
            if tf.__version__ >= '2.3':
              end_step = step + data_handler.step_increment
            else:
              end_step = step
            # if not gezi.get('tpu'):
            callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
      if batch_outputs is None:
        raise ValueError('Expect x to be a non-empty array or dataset.')
      callbacks.on_predict_end()
    with tf.device('/cpu:0'):
      all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
      return tf_utils.to_numpy_or_python_type(all_outputs)

  def custom_loop(model,
                  x,
                  custom_fn,
                  mark='predict',
                  steps=None,
                  callbacks=None,
                  desc='custom_loop',
                  verbose=0):
    outputs = None
    with model.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.DataHandler(x=x,
                                              batch_size=None,
                                              steps_per_epoch=steps,
                                              initial_epoch=0,
                                              epochs=1)
      steps = data_handler.inferred_steps

    # Container that configures and calls `tf.keras.Callback`s.
    if not isinstance(callbacks, callbacks_module.CallbackList):
      if verbose:
        logging.info(desc)
      callbacks = callbacks_module.CallbackList(callbacks,
                                                add_history=True,
                                                add_progbar=verbose != 0,
                                                model=model,
                                                verbose=verbose,
                                                epochs=1,
                                                steps=steps)
      if tf.__version__ < '2.3':
        callbacks._t_enter_batch = 0

      loop_function = model.make_custom_loop_function(custom_fn, mark)

      model._predict_counter.assign(0)
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        with data_handler.catch_stop_iteration():
          # for step in tqdm(data_handler.steps(), total=steps, ascii=True, desc=desc):
          for step in data_handler.steps():
            callbacks.on_predict_batch_begin(step)
            loop_function(iterator)
            if tf.__version__ >= '2.3':
              if data_handler.should_sync:
                context.async_wait()
            else:
              if not data_handler.inferred_steps:
                context.async_wait()
            if tf.__version__ >= '2.3':
              end_step = step + data_handler.step_increment
            else:
              end_step = step
            callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
      if batch_outputs is None:
        raise ValueError('Expect x to be a non-empty array or dataset.')
      callbacks.on_predict_end()

# https://www.kaggle.com/kentaronakanishi/tf2-0-way-to-accumulate-gradients-in-custom-loop
# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices):
  if type(grads_or_idx_slices) == tf.IndexedSlices:
    return tf.scatter_nd(
      tf.expand_dims(grads_or_idx_slices.indices, 1),
      grads_or_idx_slices.values,
      grads_or_idx_slices.dense_shape
    )
  return grads_or_idx_slices

class Model(keras.Model):

  def __init__(self, model=None, accum_step=None, interval_steps=0, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.debug = False
    self.step = 0
    self.training = False
    self.first = True
    
    if not interval_steps:
      try:
        self.interval_teps = FLAGS.inveral_steps
      except Exception:
        self.interval_steps = 0
    else:
      self.interval_steps = interval_steps
    
    self.input_ = None
    self.inputs = None

    self.feats = []
    self.embs = self.feats
    self.feats_ = collections.OrderedDict()
    self.out_keys = []
    self.eval_keys = []
    self.str_keys = []
    self.predict_function_outhook = None
    self.loop_function = None

    self.custom_eval_function = None
    self.custom_predict_function = None
    
    self.custom_metrics = OrderedDict()
    self.metric_values = {}

    if tf.__version__ < '2.3':
      self._steps_per_execution = None
      self._predict_counter = tf.Variable(0)
      
    self.model = model
    self.remove_pred = False

    # https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
    # https://www.kaggle.com/kentaronakanishi/tf2-0-way-to-accumulate-gradients-in-custom-loop
    n_gradients = accum_step if accum_step is not None else FLAGS.acc_steps
    # self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    self.n_acum_step = 0
    self.n_gradients = n_gradients
    self.gradient_accumulation = []
    # self.init_gradient_acc()
    self.gradient_acc_inited = False

  
  def _initial_gradient_accumulation(self):
    gradient_accumulation = [
          tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
          for v in self.trainable_variables
      ]
    return gradient_accumulation
  
  def init_gradient_acc(self):
    if not self.gradient_acc_inited:
      if self.n_gradients and self.n_gradients > 1:
        self.n_acum_step = 0
        self.gradient_accumulation = self._initial_gradient_accumulation()
      self.gradient_acc_inited = True
    
  def call(self, x, **kwargs):
    if self.model is None:
      return tf.zeros((1,))
    else:
      return self.model(x, **kwargs)
    
  def append_metric(self, name, val):
    self.metric_values[name] = val
    if name not in self.custom_metrics:
      self.custom_metrics[name] = melt.metrics.CurrentMetric(name)
      
  def custom_metric(self, name, val):
    self.append_metric(name, val)
    
  def metric_(self, name, val):
    self.append_metric(name, val)
    
  def scalar(self, name, val):
    self.metric_(name, val)

  def append_metrics(self, metrics):
    for name, metric in metrics.items():
      self.append_metric(name, metric)
    
  def metrics_(self, metrics):
    self.append_metrics(metrics)

  # TODO add historgram ?
  def scalars(self, metrics):
    self.metrics_(metrics)
    
  def monitor_emb(self, emb, name='emb', zero_ratio=False, training=True):
    if training:
      if self.step == 1 or not self.interval_steps or self.step % self.interval_steps == 0:
        if FLAGS.monitor_level > 1 and (not FLAGS.disable_monitor):
          res = {
            'l1': tf.reduce_mean(emb),
            'l2': tf.reduce_mean(emb * emb),
            'max': tf.reduce_max(emb),
            'min': tf.reduce_min(emb),
            'l2_0': tf.reduce_mean(emb[0] * emb[0]), 
            'l2_1': tf.reduce_mean(emb[1] * emb[1]),
            'l2_2': tf.reduce_mean(emb[2] * emb[2]), 
            'l2_-1': tf.reduce_mean(emb[-1] * emb[-1])
          }
          if zero_ratio:
            res['0ratio'] = tf.reduce_mean(tf.cast(emb == 0, tf.float32))
            res['pos_ratio'] = tf.reduce_mean(tf.cast(emb > 0, tf.float32))
          self.metrics_(gezi.dict_prefix(res, f'{name}/'))
      
  def emb_summary(self, emb, name='emb', **kwargs):
    return self.monitor_emb(emb, name, **kwargs)    

  def train_step(self, data):
    if hasattr(self, 'custom_train_step'):
      return self.custom_train_step(data)
    
    if self.n_gradients and self.n_gradients > 1:
      return self.acc_train_step(data)
    
    # TODO 如果既需要acc gradient 又需要 multi opt？
    if FLAGS.multiopt_train_step and gezi.get('optimizers'):
      return self.multiopt_train_step(data)

    return self.basic_train_step(data)
    
  
  # def custom_train_step(self, data):
  #   pass
  
  def basic_train_step(self, data):
    self.step += 1
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(y, y_pred)
    if self.step == 1 or not self.interval_steps or self.step % self.interval_steps == 0:
      for name in self.custom_metrics:
        self.custom_metrics[name].update_state(self.metric_values[name])
    self.first = False
    return {m.name: m.result() for m in self.metrics + list(self.custom_metrics.values())}

  def multiopt_train_step(self, data):
    self.step += 1
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    opt_layers_list = gezi.get('opt_layers')
    trainable_variables = []
    idxes = [0]
    for opt_layers in opt_layers_list:
      for opt_layer in opt_layers:
        for var in opt_layer.trainable_variables:
          trainable_variables.append(var)
      idxes.append(len(trainable_variables))
    assert len(trainable_variables) == len(self.trainable_variables), f'{len(trainable_variables)} {len(self.trainable_variables)}'
    gradients = tape.gradient(loss, trainable_variables)
    optimizers = gezi.get('optimizers')
    for i, optimizer in enumerate(optimizers):
      gradients_ = gradients[idxes[i]:idxes[i+1]]
      trainable_variables_ = trainable_variables[idxes[i]:idxes[i+1]]
      optimizer.apply_gradients(zip(gradients_, trainable_variables_))

    self.compiled_metrics.update_state(y, y_pred)
    if self.step == 1 or not self.interval_steps or self.step % self.interval_steps == 0:
      for name in self.custom_metrics:
        self.custom_metrics[name].update_state(self.metric_values[name])
    self.first = False
    return {m.name: m.result() for m in self.metrics + list(self.custom_metrics.values())}

  def acc_train_step(self, data):
    self.init_gradient_acc()
    self.step += 1
    # if not self.gradient_accumulation:
    #   self.gradient_accumulation = self.get_initial_gradient_accumulation()
    assert self.gradient_accumulation, 'must set gradient acc, call model.init_acc() after model has weights'
    
    x, y = data
    # Gradient Tape
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.n_acum_step += 1
    # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    for i in range(len(self.gradient_accumulation)):
      if gradients[i] is not None:
        self.gradient_accumulation[i].assign_add(gradients[i])
    self.try_apply_accu_gradients()

    if self.step == 1 or not self.interval_steps or self.step % self.interval_steps == 0:
      for name in self.custom_metrics:
        self.custom_metrics[name].update_state(self.metric_values[name])
    self.first = False
    return {m.name: m.result() for m in self.metrics + list(self.custom_metrics.values())}
  
  def try_apply_accu_gradients(self):
    if self.n_acum_step > self.n_gradients:
      self.apply_accu_gradients()

  def apply_accu_gradients(self):
    gradient_accumulation = [flat_gradients(g) / self.n_acum_step for g in self.gradient_accumulation]
    self.optimizer.apply_gradients(
        zip(gradient_accumulation, self.trainable_variables))

    self.n_acum_step = 0
    
    for i in range(len(self.gradient_accumulation)):
      self.gradient_accumulation[i].assign(
          tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

  @staticmethod
  def is_ok(name):
    ok = True
    if FLAGS.incl_feats:
      ok = False
      for feat in FLAGS.incl_feats:
        if FLAGS.re_feats:
          if re.search(feat, name):
            ok = True
        else:
          if feat == name:
            ok = True
      if not ok:
        return False
    if FLAGS.excl_feats:
      for feat in FLAGS.excl_feats:
        if FLAGS.re_feats:
          if re.search(feat, name):
            return False
        else:
          if feat == name:
            return False
    return True

  def add(self, feat, name, expand_dim=None, squeeze_dim=None):
    if not isinstance(name, str):
      tmp = name
      name = feat
      feat = tmp
    
    if expand_dim is not None:
      feat = tf.expand_dims(feat, expand_dim)
      
    if squeeze_dim is not None:
      feat = tf.expand_dims(feat, squeeze_dim)

    if Model.is_ok(name):
      self.feats_[name] = feat
      self.feats += [feat]

  def adds(self, feats, names=None):
    if names:
      feats = zip(feats, names)
    for feat, name in feats:
      self.add(feat, name)

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def get_model(self, inputs=None):
    if inputs is None:
      inputs = features2inputs(self.input_ or self.inputs)
    inputs_ = inputs.copy()
    out = self.call(inputs)
    model = keras.Model(inputs_, out, name=self.name)
    return model
  
  def custom_loss(self, loss_obj, loss_scale=1., inputs=None, model=None):
    def loss_fn(y_true, y_pred):
      y_pred = tf.cast(y_pred, tf.float32)
      loss = loss_obj(y_true, y_pred)
      loss = mt.reduce_over(loss)
      loss *= loss_scale
      return loss
    if inputs is not None or model is not None:
      return self.loss_wrapper(loss_fn, inputs, model)
    else:
      return loss_fn

  def loss_wrapper(self, loss_fn, inputs=None, model=None):

    def loss_fn_(y_true, y_pred):
      # strategy = melt.distributed.get_strategy()
      # with strategy.scope():
      args = inspect.getargspec(loss_fn).args
      kwargs = {}
      if 'x' in args:
        kwargs['x'] = inputs or self.input_ or self.inputs
      if 'input' in args:
        kwargs['input'] = inputs or self.input_ or self.inputs
      if 'model' in args:
        kwargs['model'] = model or self

      # 注意 https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb#scrollTo=gX975dMSNw0e
      # 自定义custom loop 不使用model.fit 或者 model.fit train_step自定义 不使用compiled_loss的时候 按照上面说明 是loss fn希望 / global_batch_size的
      # 但是默认keras loss 应该是按照per gpu自己内部 / rplica_batch_size的 然后再model.fit compiled_loss使用的时候会自动给你 / num_replicas达到这个效果 注意metric loss会再 * num_replicas转回来
      # 所以keras的 就 / replica_batch_size就好了
      return loss_fn(y_true, y_pred, **kwargs)

    return loss_fn_

  def get_loss(self, loss_fn=None):
    if loss_fn is not None:
      ic(loss_fn)
      return self.loss_wrapper(loss_fn)
    else:
      ic(self.get_loss_fn())
      return self.loss_wrapper(self.get_loss_fn())
      

  def get_loss_fn(self):
    raise NotImplementedError

  def input_hook(self, data, keys=[]):
    data = data_adapter.expand_1d(data)
    x, y, _ = data_adapter.unpack_x_y_sample_weight(data)
    m = {}
    keys = keys or self.eval_keys
    if not keys:
      keys = ['id']
    for key in keys:
      if isinstance(x, dict):
        if key in x:
          m[key] = x[key]
      elif isinstance(x, (list, tuple)):
        for x_ in x:
          if isinstance(x_, dict):
            if key in x_:
              m[key] = x_[key]
              break
    if (y is not None) and (not self.remove_pred):
      m['y'] = y
    return m

  def out_hook(self, keys=[]):
    m = {}
    keys = keys or self.out_keys
    for key in keys:
      if hasattr(self, key):
        m[key] = getattr(self, key)
    return m

  def merge(self, feats):
    for feat in feats:
      self.feats_[feat] = feats[feat]
      self.feats.append(feats[feat])

  def print_feats(self, print_fn=logging.debug):
    if not hasattr(self, 'print_feats_'):
      self.print_feats_ = False
    if self.first and (not self.print_feats_):
      print_fn(f'Num model features {len(self.feats_)}:')
      for i, (feat, feat_val) in enumerate(self.feats_.items()):
        print_fn(i, feat, feat_val.shape)
    self.print_feats_ = True

  def clear(self):
    self.feats = []
    self.feats_ = collections.OrderedDict()

  def decode(self, res):
    if isinstance(res, dict):
      for key in self.eval_keys:
        if key in res:
          res[key] = gezi.squeeze(res[key])
      for key in self.str_keys:
        if key in res:
          res[key] = gezi.decode(res[key])
    return res

  def infer(self,
            x,
            steps=None,
            dump_inputs=True,
            callbacks=None,
            write_fn=None,
            desc='predict',
            verbose=0,
            leave=True):
    if not callbacks and not verbose:
      callbacks = [TQDMProgressBar(desc, leave=leave)]
    res = predict(self, x, steps, dump_inputs, callbacks, write_fn, desc,
                  verbose)
    res = self.decode(res)
    return res

  def custom_predict(self,
                     x,
                     steps=None,
                     dump_inputs=True,
                     callbacks=None,
                     desc='predict',
                     verbose=0,
                     leave=True):
    if not callbacks and not verbose:
      callbacks = [TQDMProgressBar(desc, leave=leave)]
    res = predict(self, x, steps, dump_inputs, callbacks, desc, verbose)
    res = self.decode(res)
    return res

  def loop(self,
           x,
           steps=None,
           callbacks=None,
           desc='loop',
           verbose=0,
           leave=True):
    if not callbacks and not verbose:
      callbacks = [TQDMProgressBar(desc, leave=leave)]
    res = loop(self, x, steps, callbacks, desc, verbose)
    res = self.decode(res)
    return res

  def make_predict_outhook_function(self, dump_inputs=True):
    """Creates a function that executes one step of inference.
    This method can be overridden to support custom inference logic.
    This method is called by `Model.predict` and `Model.predict_on_batch`.
    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.predict_step`.
    This function is cached the first time `Model.predict` or
    `Model.predict_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.
    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return the outputs of the `Model`.
    """
    from tensorflow.python.keras.engine.training import reduce_per_replica, def_function

    if self.predict_function_outhook is not None:
      return self.predict_function_outhook

    def step_function(model, iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        inputs = model.input_hook(data) if dump_inputs else {}
        outputs = model.predict_step(data)
        others = model.out_hook()

        # ic(inputs)
        # ic(oututs)
        # ic(others)

        if inputs or others:
          if not isinstance(outputs, dict):
            if not self.remove_pred:
              outputs = {'pred': outputs}
            else:
              outputs = {'pred': [0.]}
          outputs.update(inputs)
          outputs.update(others)

        # Ensure counter is updated only if `test_step` succeeds.
        if tf.__version__ >= '2.3':
          with tf.control_dependencies(_minimum_control_deps(outputs)):
            model._predict_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      # ic(outputs)
      outputs = reduce_per_replica(outputs,
                                   self.distribute_strategy,
                                   reduction='concat')
      return outputs

    if (self._steps_per_execution is None or
        self._steps_per_execution.numpy().item() == 1):

      def predict_function(iterator):
        """Runs an evaluation execution with one step."""
        return step_function(self, iterator)

    else:

      def predict_function(iterator):
        """Runs an evaluation execution with multiple steps."""
        outputs = step_function(self, iterator)
        for _ in math_ops.range(self._steps_per_execution - 1):
          directives.set_loop_options(
              shape_invariants=[(
                  t, tf_utils.get_tensor_spec(t, dynamic_batch=True).shape)
                                for t in nest.flatten(outputs)])
          step_outputs = step_function(self, iterator)
          outputs = nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs,
                                       step_outputs)
        return outputs

    if not self.run_eagerly:
      predict_function = def_function.function(predict_function,
                                               experimental_relax_shapes=True)

    self.predict_function_outhook = predict_function
    return self.predict_function_outhook

  def make_loop_function(self):
    """Creates a function that executes one step of inference.
    This method can be overridden to support custom inference logic.
    This method is called by `Model.predict` and `Model.predict_on_batch`.
    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.predict_step`.
    This function is cached the first time `Model.predict` or
    `Model.predict_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.
    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return the outputs of the `Model`.
    """
    from tensorflow.python.keras.engine.training import reduce_per_replica, def_function

    if self.loop_function is not None:
      return self.loop_function

    def step_function(model, iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        outputs = model.input_hook(data)

        # Ensure counter is updated only if `test_step` succeeds.
        if tf.__version__ >= '2.3':
          with tf.control_dependencies(_minimum_control_deps(outputs)):
            model._predict_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = reduce_per_replica(outputs,
                                   self.distribute_strategy,
                                   reduction='concat')
      return outputs

    if (self._steps_per_execution is None or
        self._steps_per_execution.numpy().item() == 1):

      def predict_function(iterator):
        """Runs an evaluation execution with one step."""
        return step_function(self, iterator)

    else:

      def predict_function(iterator):
        """Runs an evaluation execution with multiple steps."""
        outputs = step_function(self, iterator)
        for _ in math_ops.range(self._steps_per_execution - 1):
          directives.set_loop_options(
              shape_invariants=[(
                  t, tf_utils.get_tensor_spec(t, dynamic_batch=True).shape)
                                for t in nest.flatten(outputs)])
          step_outputs = step_function(self, iterator)
          outputs = nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs,
                                       step_outputs)
        return outputs

    if not self.run_eagerly:
      predict_function = def_function.function(predict_function,
                                               experimental_relax_shapes=True)

    self.loop_function = predict_function
    return self.loop_function

  def make_custom_loop_function(self,
                                custom_fn,
                                dump_inputs=True,
                                mark='predict'):
    """Creates a function that executes one step of inference.
    This method can be overridden to support custom inference logic.
    This method is called by `Model.predict` and `Model.predict_on_batch`.
    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.predict_step`.
    This function is cached the first time `Model.predict` or
    `Model.predict_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.
    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return the outputs of the `Model`.
    """
    from tensorflow.python.keras.engine.training import reduce_per_replica, def_function

    if mark == 'predict':
      if self.custom_predict_function is not None:
        return self.custom_predict_function
    elif mark == 'eval':
      if self.custom_eval_function is not None:
        return self.custom_eval_function
    else:
      raise ValueError(mark)

    def step_function(model, iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        inputs = model.input_hook(data) if dump_inputs else {}
        outputs = model.predict_step(data)
        others = model.out_hook()
        custom_fn(model, inputs, outputs)

        # Ensure counter is updated only if `test_step` succeeds.
        if tf.__version__ >= '2.3':
          with tf.control_dependencies(_minimum_control_deps(outputs)):
            model._predict_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = reduce_per_replica(outputs,
                                   self.distribute_strategy,
                                   reduction='concat')
      return outputs

    if (self._steps_per_execution is None or
        self._steps_per_execution.numpy().item() == 1):

      def predict_function(iterator):
        """Runs an evaluation execution with one step."""
        return step_function(self, iterator)

    else:

      def predict_function(iterator):
        """Runs an evaluation execution with multiple steps."""
        outputs = step_function(self, iterator)
        for _ in math_ops.range(self._steps_per_execution - 1):
          directives.set_loop_options(
              shape_invariants=[(
                  t, tf_utils.get_tensor_spec(t, dynamic_batch=True).shape)
                                for t in nest.flatten(outputs)])
          step_outputs = step_function(self, iterator)
          outputs = nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs,
                                       step_outputs)
        return outputs

    if not self.run_eagerly:
      predict_function = def_function.function(predict_function,
                                               experimental_relax_shapes=True)

    if mark == 'predict':
      self.custom_predict_function = predict_function
      return self.custom_predict_function
    else:
      self.custom_eval_function = predict_function
      return self.custom_eval_function


def exists_model(model_dir):
  return os.path.exists(model_dir) and (not os.path.isdir(model_dir) or
                                        glob.glob(model_dir + '/model*ckpt*'))


def adjust_lrs(x, ratio=None, name='learning_rate_weights'):
  if ratio is None:
    ratios = tf.compat.v1.get_collection(name)[-1]
    x = x * ratios + tf.stop_gradient(x) * (1 - ratios)
  else:
    x = x * ratio + tf.stop_gradient(x) * (1 - ratio)
  return x


def adjust_weights(x, ratio=None, name='learning_rate_weights'):
  if ratio is None:
    ratios = tf.compat.v1.get_collection(name)[-1]
    x = x * ratios
  else:
    x = x * ratio
  return x


def try_convert_images(images):
  if not isinstance(images, (list, tuple, np.ndarray)):
    images = [images]
  if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
    images = [melt.image.read_image(image) for image in images]
  return images


def freeze_graph(sess,
                 model_path,
                 global_step=None,
                 output_collection_names=None,
                 output_node_names=None,
                 ignores=[]):
  if output_collection_names is None and output_node_names is None:
    return None

  graph_def = sess.graph_def
  graph = sess.graph

  if global_step is not None:  # allow 0
    outfile = '%s-%d.pb' % (model_path, global_step)
    outmapfile = '%s-%d.map' % (model_path, global_step)
  else:
    outfile = '%s.pb' % model_path
    outmapfile = '%s.map' % model_path

  ignores = ['loss', 'iterator', 'saveable', 'global_step', 'learning_rate'
            ] + ignores

  def will_ignore(x):
    if isinstance(x, dict):
      return False
    if not hasattr(x, 'name'):
      return False
    for ignore in ignores:
      if ignore in x.name:
        return True
    return False

  if output_node_names is None:
    from tempfile import NamedTemporaryFile
    output_node_names = []
    outmap = open(outmapfile, 'w')
    for cname in output_collection_names:
      for item in graph.get_collection(cname):
        # [TopKV2(values=<tf.Tensor 'TopKV2_1:0' shape=(1,) dtype=int32>, indices=<tf.Tensor 'TopKV2_1:1' shape=(1,) dtype=int32>)]
        # f.get_collection('y')[0][0].name Out[17]: u'TopKV2_1:0'

        # bypass :0
        # if not hasattr(item, 'name'):
        #   print('item no name for', item, file=sys.stderr)
        #   continue
        # :1 for like top_k, :2 for future usage might length 3 tuple
        #logging.debug('freeze_graph:', cname, item.name)
        if item is None:
          continue
        if will_ignore(item):
          continue
        if not hasattr(item, 'name'):
          continue
        if item.name.endswith(':0') or item.name.endswith(
            ':1') or item.name.endswith(':2'):
          opname = item.name[:-2]
        else:
          if 'initializer' in cname:
            opname = item.name
          else:
            continue
        output_node_names.append(opname)
        #logging.debug('freeze_graph filter:', cname, item.name)
        print('%s\t%s' % (cname, item.name), file=outmap)

  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess, graph_def, output_node_names)

  logging.info('freeze graph to', outfile)
  tf.train.write_graph(frozen_graph_def,
                       os.path.dirname(outfile),
                       os.path.basename(outfile),
                       as_text=False)
  # with tf.io.gfile.GFile (outfile, "wb") as f:
  #   f.write(frozen_graph_def.SerializeToString())
  # tf.train.export_meta_graph(
  #   filename=outfile,
  #   graph_def=frozen_graph_def,
  #   collection_list=output_collection_names)

  model_dir = os.path.dirname(outfile)
  #print('model_dir', model_dir)
  maps = glob.glob('%s/*.map' % model_dir)
  #print('maps', maps)
  for map_ in maps:
    if map_.endswith('model.map'):
      continue
    index_file = map_.replace('.map', '.index')
    if not os.path.exists(index_file):
      pb_file = map_.replace('.map', '.pb')
      logging.info('remove %s %s' % (map_, pb_file))
      os.remove(map_)
      os.remove(pb_file)

  return frozen_graph_def


def is_raw_image(image_features):
  return isinstance(image_features[0], np.string_)


def set_learning_rate(lr, sess=None, name='learning_rate'):
  if not sess:
    sess = melt.get_session()
  sess.run(tf.assign(tf.get_collection(name)[-1], lr))


def multiply_learning_rate(lr, sess=None, name='learning_rate'):
  if not sess:
    sess = melt.get_session()
  sess.run(
      tf.assign(tf.get_collection(name)[-1],
                tf.get_collection(name)[-1] * lr))


#https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
def init_uninitialized_variables(sess, list_of_variables=None):
  if list_of_variables is None:
    list_of_variables = tf.global_variables()
  uninitialized_variables = list(
      tf.get_variable(name) for name in sess.run(
          tf.report_uninitialized_variables(list_of_variables)))
  uninitialized_variables = tf.group(uninitialized_variables,
                                     tf.local_variables_initializer())
  sess.run(tf.variables_initializer(uninitialized_variables))
  return uninitialized_variables


def get_global_step_(model_dir, num_steps_per_epoch, fix_step=True):
  if not model_dir:
    return 0

  checkpoint_path = get_model_path(model_dir)
  if os.path.isdir(checkpoint_path) or not os.path.exists(checkpoint_path +
                                                          '.index'):
    return 0

  pre_step = get_model_step(checkpoint_path)
  if not num_steps_per_epoch or not fix_step:
    return pre_step

  pre_epoch = melt.get_model_epoch(checkpoint_path)
  if pre_epoch is None:
    return pre_step

  fixed_pre_step = pre_step
  if abs(pre_step / num_steps_per_epoch - pre_epoch) > 0.1:
    fixed_pre_step = int(pre_epoch * num_steps_per_epoch)
    return fixed_pre_step
  else:
    return pre_step


def checkpoint_exists(checkpoint_path):
  return not os.path.isdir(checkpoint_path) and \
         os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path + '.index')


def get_checkpoint_varnames(model_dir):
  checkpoint_path = get_model_path(model_dir)
  #if model_dir is dir then checkpoint_path should be model path not model dir like
  #/home/gezi/new/temp/image-caption/makeup/model/bow.inceptionResnetV2.finetune.later/model.ckpt-589.5-1011000
  #if user input model_dir is like above model path we assume it to be correct and exists not check!
  if not checkpoint_path or not os.path.exists(checkpoint_path + '.index'):
    return None
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    varnames = [var_name for var_name in var_to_shape_map]
    return varnames
  except Exception:
    print(traceback.format_exc())
    return None


def varname_in_checkpoint(varname_part, model_dir, mode='in'):
  assert varname_part
  varnames = get_checkpoint_varnames(model_dir)
  if not varnames:
    return False
  else:
    varnames_exists = False
    for varname in varnames:
      if mode == 'in':
        if varname_part in varname:
          varnames_exists = True
          break
      elif mode == 'startswith':
        if varname.startswith(varname_part):
          varnames_exists = True
          break
      elif mode == 'exact_match':
        if varname_part == varname:
          varnames_exists = True

    return varnames_exists


def has_image_model(model_dir, image_model_name):
  return varname_in_checkpoint(image_model_name, model_dir)


def try_add_to_collection(name, op):
  if not tf.get_collection(name):
    tf.add_to_collection(name, op)


def remove_from_collection(key):
  #must use ref get list and set to empty using [:] = [] or py3 can .clear
  #https://stackoverflow.com/questions/850795/different-ways-of-clearing-lists
  l = tf.get_collection_ref(key)
  l[:] = []


def rename_from_collection(key, to_key, index=0, scope=None):
  l = tf.get_collection_ref(key)
  if l:
    tf.add_to_collection(to_key, l[index])
    l[:] = []


#https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
def initialize_uninitialized_vars(sess):
  import itertools
  from itertools import compress
  global_vars = tf.global_variables()
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                  for var in global_vars])
  not_initialized_vars = list(compress(global_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))


#In [3]: tf.contrib.layers.OPTIMIZER_CLS_NAMES
#Out[3]:
#{'Adagrad': tensorflow.python.training.adagrad.AdagradOptimizer,
# 'Adam': tensorflow.python.training.adam.AdamOptimizer,
# 'Ftrl': tensorflow.python.training.ftrl.FtrlOptimizer,
# 'Momentum': tensorflow.python.training.momentum.MomentumOptimizer,
# 'RMSProp': tensorflow.python.training.rmsprop.RMSPropOptimizer,
# 'SGD': tensorflow.python.training.gradient_descent.GradientDescentOptimizer}

optimizers = {
    'grad':
        tf.compat.v1.train.GradientDescentOptimizer,
    'sgd':
        tf.compat.v1.train.GradientDescentOptimizer,
    'adagrad':
        tf.compat.v1.train.AdagradOptimizer,
    # TODO notice tensorflow adagrad no epsilon param..  See if kears optimizers better ?
    #'adagrad': lambda lr: tf.train.AdagradOptimizer(lr, epsilon=1e-06), # keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
    #'adam': tf.train.AdamOptimizer,
    #'adam': lambda lr: tf.train.AdamOptimizer(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08), #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    'adam':
        tf.compat.v1.train.AdamOptimizer,
    'Adam':
        tf.compat.v1.train.AdamOptimizer,
    'adam_t2t':
        lambda lr: tf.compat.v1.train.AdamOptimizer(
            lr, epsilon=1e-06, beta1=0.85, beta2=0.997),
    #'adadelta': tf.train.AdadeltaOptimizer
    'adadelta':
        lambda lr: tf.compat.v1.train.AdadeltaOptimizer(
            lr, epsilon=1e-6
        ),  #follow squad, also keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    # # still not fix https://github.com/tensorflow/tensorflow/pull/15665
    # # 'nadam': tf.contrib.opt.NadamOptimizer, # TODO FIXME nadam not work, InvalidArgumentError (see above for traceback): Incompatible shapes: [2737,300] vs. [91677,300]
    # #'momentum': lambda lr, momentum: tf.train.MomentumOptimizer(lr, momentum=momentum) # in melt.app.train
    # #'adamax': tf.contrib.opt.AdaMaxOptimizer, # will got NAN ...
    # #'adamax': lambda lr: tf.contrib.opt.AdaMaxOptimizer(lr, epsilon=1e-8),
    # 'adamax': melt.training.adamax.AdaMaxOptimizer,
    # 'ftrl': tf.train.FtrlOptimizer,
    # 'lazyadam': tf.contrib.opt.LazyAdamOptimizer,
    # #'adamax': tf.keras.optimizers.Adamax,  # tf can not directly use kears optimzier...
    # 'proximaladagrad': lambda lr: tf.train.ProximalAdagradOptimizer(learning_rate=lr,l1_regularization_strength=0.001, l2_regularization_strength=0.001),
    # 'adamw': tf.contrib.opt.AdamWOptimizer,
    # 'lazyadamw': tf.contrib.opt.extend_with_decoupled_weight_decay(tf.contrib.opt.LazyAdamOptimizer),
}

if tf.__version__ < '2':
  # TODO maybe try opt.LARSOptimizer opt.LazyAdamGSOptimizer
  optimizers['lazyadam'] = tf.contrib.opt.LazyAdamOptimizer
  optimizers['LazyAdam'] = tf.contrib.opt.LazyAdamOptimizer
  optimizers['lars'] = tf.contrib.opt.LARSOptimizer
  optimizers['lazyadamgs'] = tf.contrib.opt.LazyAdamGSOptimizer
else:
  try:
    import tensorflow_addons as tfa
    optimizers['lazyadam'] = tfa.optimizers.LazyAdam
  except Exception:
    pass

keras_optimizers = {
    'adagrad': tf.keras.optimizers.Adagrad,
    'adam': tf.keras.optimizers.Adam,
    'adadelta': tf.keras.optimizers.Adadelta,
    'nadam': tf.keras.optimizers.Nadam,
}


def get_optimizer(name):
  if not isinstance(name, str):
    return name
  # TODO how to use keras optimizers especially like nadam ? seems different api
  # if name.lower() in keras_optimizers:
  #   return keras_optimizers[name.lower()]
  # elif name.lower() in optimizers:
  if name.lower() in optimizers:
    return optimizers[name.lower()]
  else:
    try:
      return tf.contrib.layers.OPTIMIZER_CLS_NAMES[name]
    except Exception:
      return getattr(tf.compat.v1.train, name)
  # if name in tf.contrib.layers.OPTIMIZER_CLS_NAMES:
  #   return tf.contrib.layers.OPTIMIZER_CLS_NAMES[name]
  # else:
  #   return optimizers[name.lower()]


def get_session(
    allow_growth=True,
    log_device_placement=False,
    allow_soft_placement=True,
    debug=False,
    # enable_xla=False,
    device_count=None,
    gpus=None,
    graph=None):
  """
  TODO FIXME get_session will casue  at last
#Exception UnboundLocalError: "local variable 'status' referenced before assignment" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x858af10>> ignored
#TRACE: 03-17 08:22:26:   * 0 [clear]: tag init stat error

global or inside function global sess will cause this but not big problem for convenience just accpet right now
  """
  if not hasattr(get_session, 'sess') or get_session.sess is None:
    if device_count is None:
      config = tf.compat.v1.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
    else:
      config = tf.compat.v1.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement,
          device_count=device_count)
    config.gpu_options.allow_growth = allow_growth

    if FLAGS.enable_xla:
      from tensorflow.core.protobuf import rewriter_config_pb2
      config = tf.compat.v1.ConfigProto()
      tf.config.optimizer.set_jit(True)
      config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
      # config.graph_options.optimizer_options.global_jit_level = (
      #     tf.compat.v1.OptimizerOptions.ON_2)
      # Disable PinToHostOptimizer in grappler when enabling XLA because it causes
      # OOM and performance regression.
      config.graph_options.rewrite_options.pin_to_host_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)

    dist = gezi.DistributedWrapper()
    distributed = dist.is_distributed()
    if distributed:
      local_rank = dist.local_rank()
      config.gpu_options.visible_device_list = str(local_rank)
      # sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
      #                                          config=config)
    else:
      # print(gpus)
      # print(tf.config.experimental.list_physical_devices('GPU'))
      if gpus:
        config.gpu_options.visible_device_list = ','.join(map(str, gpus))

    gezi.set('tf_config', config)

    #config.operation_timeout_in_ms=600000
    #NOTICE https://github.com/tensorflow/tensorflow/issues/2130 but 5000 will cause init problem!
    #config.operation_timeout_in_ms=50000   # terminate on long hangs
    #https://github.com/tensorflow/tensorflow/issues/2292 allow_soft_placement=True
    use_tpu = False
    try:
      use_tpu = FLAGS.use_tpu
    except Exception:
      pass
    if use_tpu:
      tpu_cluster_resolver = None
      if FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
      get_session.sess = tf.compat.v1.Session(tpu_cluster_resolver)
    elif FLAGS.ps_strategy and FLAGS.ps_hosts:
      # return None
      config = gezi.get('tf_config')
      assert FLAGS.worker_hosts, 'set FLAGS.worker_hosts or FLAGS.num_gpus > 1'
      cluster = tf.train.ClusterSpec({
          'ps': FLAGS.ps_hosts.split(','),
          'worker': FLAGS.worker_hosts.split(','),
      })

      server = tf.train.Server(cluster,
                               job_name=FLAGS.job_name,
                               task_index=FLAGS.task_index,
                               config=config)

      if FLAGS.job_name == 'ps':
        logging.info('ps running on', FLAGS.ps_hosts)
        fp = gezi.get('fp')
        if fp:
          fcntl.flock(fp, fcntl.LOCK_UN)
          fp.close()
          fp = None
        return server.join()

      device_setter = tf.train.replica_device_setter(
          worker_device='/job:worker/task:%d' % FLAGS.task_index,
          ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
              len(FLAGS.ps_hosts.split(',')),
              tf.contrib.training.byte_size_load_fn),
          cluster=cluster)

      # return None

      get_session.sess = tf.compat.v1.Session(config=config, graph=graph)

      # with tf.device(device_setter):
      #   global_step = tf.train.get_or_create_global_step()
      #   get_session.sess = tf.train.MonitoredTrainingSession(master=server.target,
      #                                             is_chief=(FLAGS.task_index == 0),
      #                                             checkpoint_dir=FLAGS.log_dir)
      gezi.set('device_setter', device_setter)
      gezi.set('server', server)
    else:
      server = None
      try:
        device_setter = tf.train.replica_device_setter(0)
      except Exception:
        device_setter = ''
      gezi.set('device_setter', device_setter)
      gezi.set('server', server)
      get_session.sess = tf.compat.v1.Session(config=config, graph=graph)

    assert tf.__version__ < '2' or FLAGS.graph
    K.set_session(get_session.sess)

    if debug:
      from tensorflow.python import debug as tf_debug
      get_session.sess = tf_debug.LocalCLIDebugWrapperSession(get_session.sess)

  return get_session.sess


def gen_session(graph=None,
                log_device_placement=False,
                allow_soft_placement=True,
                debug=False):
  config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                          log_device_placement=log_device_placement)
  sess = tf.Session(config=config, graph=graph)
  if debug:
    from tensorflow.python import debug as tf_debug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  return sess


#def get_session(log_device_placement=False, allow_soft_placement=True, debug=False):
#  """
#  TODO FIXME get_session will casue  at last
##Exception UnboundLocalError: "local variable 'status' referenced before assignment" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x858af10>> ignored
##TRACE: 03-17 08:22:26:   * 0 [clear]: tag init stat error

#global or inside function global sess will cause this but not big problem for convenience just accpet right now
#  """
#  if not hasattr(get_session, 'sess') or get_session.sess is None:
#    config=tf.ConfigProto(
#      allow_soft_placement=allow_soft_placement,
#      log_device_placement=log_device_placement)
#    #config.operation_timeout_in_ms=600000
#    #NOTICE https://github.com/tensorflow/tensorflow/issues/2130 but 5000 will cause init problem!
#    #config.operation_timeout_in_ms=50000   # terminate on long hangs
#    sess = tf.Session(config=config)
#    if debug:
#      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#  return sess


def gen_train_op(loss,
                 learning_rate,
                 optimizer=tf.compat.v1.train.AdagradOptimizer):
  train_op = optimizer(learning_rate).minimize(loss)
  return train_op


def gen_train_op_byname(loss, learning_rate, name='adagrad'):
  optimizer = optimizers.get(name.lower(), tf.train.AdagradOptimizer)
  train_op = optimizer(learning_rate).minimize(loss)
  return train_op


#TODO add name, notice if num_gpus=1 is same as num_gpus=0
#but for num_gpus=0 we will not consider multi gpu mode
#so num_gpus=1 will not use much, just for mlti gpu test purpose
#from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py def train()

# def tower(loss_function, num_gpus=1, training=True, name=''):
#   towers = []
#   update_ops = []
#   for i in range(num_gpus):
#     with tf.device('/gpu:%d' % i):
#       #print(tf.get_variable_scope().reuse)
#       with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
#         with tf.name_scope('%s_%d' % ('tower', i)) as name_scope:
#           if 'i' in inspect.getargspec(loss_function).args:
#             loss = loss_function(i)
#           else:
#             loss = loss_function()
#           # Reuse variables for the next tower. itersting.. not work.. for cifar10 ._conv...
#           #print(tf.get_variable_scope().reuse)
#           #tf.get_variable_scope().reuse_variables()
#           #print(tf.get_variable_scope().reuse)
#           # REMIND actually for training other metrics like acc... will only record the last one, I think this is enough!
#           if isinstance(loss, (list, tuple)) and training:
#             loss = loss[0]
#           towers.append(loss)
#           if i == 0 and training:
#             # Only trigger batch_norm moving mean and variance update from
#             # the 1st tower. Ideally, we should grab the updates from all
#             # towers but these stats accumulate extremely fast so we can
#             # ignore the other stats from the other towers without
#             # significant detriment.
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
#                                             name_scope)
#   if training:
#     return towers, update_ops
#   else:
#     towers = [list(x) if isinstance(x, tuple) else x for x in towers]
#     return towers


# TODO will this be ok.. ?
def tower(loss_function, num_gpus=1, training=True, name=''):
  towers = []
  update_ops = []
  for i in range(num_gpus):
    # device = 'GPU' if not FLAGS.enable_xla else 'XLA_GPU'
    device = 'gpu'
    with tf.device(f'/{device}:{i}'):
      #print(tf.get_variable_scope().reuse)
      with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                       reuse=bool(i != 0)):
        #with tf.name_scope('%s_%d' % ('tower', i)) as name_scope:
        if 'i' in inspect.getargspec(loss_function).args:
          loss = loss_function(i)
        else:
          loss = loss_function()
        # Reuse variables for the next tower. itersting.. not work.. for cifar10 ._conv...
        #print(tf.get_variable_scope().reuse)
        #tf.get_variable_scope().reuse_variables()
        #print(tf.get_variable_scope().reuse)
        # REMIND actually for training other metrics like acc... will only record the last one, I think this is enough!
        if isinstance(loss, (list, tuple)) and training:
          loss = loss[0]
        towers.append(loss)
        if i == 0 and training:
          # Only trigger batch_norm moving mean and variance update from
          # the 1st tower. Ideally, we should grab the updates from all
          # towers but these stats accumulate extremely fast so we can
          # ignore the other stats from the other towers without
          # significant detriment.
          # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
          #                                name_scope)
          update_ops = tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.UPDATE_OPS)
  if training:
    return towers, update_ops
  else:
    towers = [list(x) if isinstance(x, tuple) else x for x in towers]
    return towers


tower_losses = tower


# from cifar10_estimator example code
# TODO can it be used with out input of batch_size so as can be used for buckets length ? different batch size how to ?
def _split_batch(batch_datas, batch_size, num_shards, training=True):
  #with tf.device('/cpu:0'):
  batch_datas = [
      tf.unstack(batch_data, num=batch_size, axis=0)
      for batch_data in batch_datas
  ]

  new_batch_datas = []
  for i in range(len(batch_datas)):
    new_batch_datas.append([[] for i in range(num_shards)])

  batch_size_per_gpu = batch_size // num_shards
  assert batch_size == batch_size_per_gpu * num_shards

  for i in range(batch_size):
    idx = i % num_shards if training else i // batch_size_per_gpu
    for j in range(len(batch_datas)):
      new_batch_datas[j][idx].append(batch_datas[j][i])

  def stack(x):
    try:
      return tf.parallel_stack(x)
    except Exception:
      return tf.stack(x)

  for i in range(len(batch_datas)):
    #new_batch_datas[i] = [tf.parallel_stack(x) for x in new_batch_datas[i] if x]
    new_batch_datas[i] = [stack(x) for x in new_batch_datas[i] if x]

  return tuple(new_batch_datas)


def split_batch(batch_datas, batch_size, num_shards, training=True):
  with tf.device('/cpu:0'):
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return tuple([x] for x in batch_datas)

    if not isinstance(batch_datas[0], dict):
      return _split_batch(batch_datas, batch_size, num_shards, training)
    else:
      # x, y (x is dict, y not)
      assert len(batch_datas) == 2
      keys = batch_datas[0].keys()
      #batch_datas = [batch_datas[0][key] for key in keys] + [batch_datas[-1]]
      batch_datas = list(batch_datas[0].values()) + [batch_datas[-1]]
      batch_datas = _split_batch(batch_datas, batch_size, num_shards, training)
      # print(batch_datas)
      # TODO... why append ok... x = [{}] * num_shards not ok..
      # x = [{}] * num_shards
      x = []
      for j in range(num_shards):
        m = {}
        for i, key in enumerate(keys):
          #x[j][key] = batch_datas[i][j]
          m[key] = batch_datas[i][j]
        x.append(m)

      y = batch_datas[-1]
      return x, y

      # for i, key in enumerate(keys):
      #   for j in range(num_shards):
      #     x[j][key] = batch_datas[i][j]
      # y = batch_datas[-1]
      # print('-----------x', x)
      # print('-----------y', y)
      # return x, y
      #return batch_datas


def is_cudnn_cell(cell):
  return isinstance(
      cell, (tf.contrib.cudnn_rnn.CudnnGRU, tf.contrib.cudnn_rnn.CudnnLSTM))


# # TODO now for hadoop can only run tf 1.2
# try:
#   rnn_cells = {
#     'basic_lstm': tf.contrib.rnn.BasicLSTMCell,
#     'lstm': tf.contrib.rnn.LSTMCell, #LSTMCell is faster then BasicLSTMCell
#     'gru': tf.contrib.rnn.GRUCell,
#     'lstm_block': tf.contrib.rnn.LSTMBlockCell, #LSTMBlockCell is faster then LSTMCell
#     'lstm_block_fused': tf.contrib.rnn.LSTMBlockFusedCell,
#     'cudnn_lstm': tf.contrib.cudnn_rnn.CudnnLSTM,
#     'cudnn_gru': tf.contrib.cudnn_rnn.CudnnGRU,
#     'cudnn_compat_lstm': tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell,
#     'cudnn_compat_gru': tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
#     }
# except Exception:
#   rnn_cells = {
#     'basic_lstm': tf.contrib.rnn.BasicLSTMCell,
#     'lstm': tf.contrib.rnn.LSTMCell, #LSTMCell is faster then BasicLSTMCell
#     'gru': tf.contrib.rnn.GRUCell,
#     'lstm_block': tf.contrib.rnn.LSTMBlockCell, #LSTMBlockCell is faster then LSTMCell
#     'lstm_block_fused': tf.contrib.rnn.LSTMBlockFusedCell,
#     }


#from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py lstm_cell()
def create_rnn_cell(num_units,
                    is_training=True,
                    initializer=None,
                    forget_bias=1.0,
                    num_layers=1,
                    keep_prob=1.0,
                    input_keep_prob=1.0,
                    Cell=None,
                    cell_type='lstm',
                    scope=None):
  with tf.variable_scope(scope or 'create_rnn_cell') as scope:
    if initializer:
      scope.set_initializer(initializer)
    if Cell is None:
      Cell = rnn_cells.get(cell_type.lower(), tf.contrib.rnn.LSTMCell)
      print('cell:', Cell, file=sys.stderr)

    if Cell is tf.contrib.cudnn_rnn.CudnnGRU or Cell is tf.contrib.cudnn_rnn.CudnnLSTM:
      cell = Cell(num_layers=num_layers,
                  num_units=num_units,
                  dropout=(1. - keep_prob))
      return cell

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def cell_():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(Cell.__init__).args:
        if 'forget_bias' in inspect.getargspec(Cell.__init__).args:
          return Cell(num_units,
                      forget_bias=forget_bias,
                      reuse=tf.get_variable_scope().reuse)
        else:
          return Cell(num_units, reuse=tf.get_variable_scope().reuse)
      else:
        if 'state_is_tuple' in inspect.getargspec(Cell.__init__).args:
          if 'forget_bias' in inspect.getargspec(Cell.__init__).args:
            return Cell(num_units, forget_bias=forget_bias, state_is_tuple=True)
          else:
            return Cell(num_units, state_is_tuple=True)
        else:
          if 'forget_bias' in inspect.getargspec(Cell.__init__).args:
            return Cell(num_units, forget_bias=forget_bias)
          else:
            return Cell(num_units)

    attn_cell = cell_
    if is_training and (keep_prob < 1 or input_keep_prob < 1):

      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(cell_(),
                                             input_keep_prob=input_keep_prob,
                                             output_keep_prob=keep_prob)

    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell(
          [attn_cell() for _ in range(num_layers)], state_is_tuple=True)
    else:
      cell = attn_cell()
    #--now cell share graph by default so below is wrong.. will share cell for each layer
    ##cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    return cell


def unpack_cell(cell):
  """Unpack the cells because the stack_bidirectional_dynamic_rnn
  expects a list of cells, one per layer."""
  if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
    return cell._cells  #pylint: disable=W0212
  else:
    return [cell]


#-------for train flow
def show_precision_at_k(result, k=1):
  if len(result) == 1:
    accuracy = result
    print('precision@%d:' % k, '%.3f' % accuracy)
  else:
    loss = result[0]
    accuracy = result[1]
    print('loss:', '%.3f' % loss, 'precision@%d:' % k, '%.3f' % accuracy)


def print_results(results, names=None):
  """
  standard result print
  """
  results = gezi.get_singles(results)
  if names is None:
    print(gezi.pretty_floats(results))
  else:
    if len(names) == len(results) - 1:
      names.insert(0, 'loss')
    # if len(names) == 1 and names[0] == 'loss':
    #   names[0] = ''
    if len(names) == len(results):
      print(','.join(gezi.get_value_name_list(results, names)))
    else:
      print(','.join(gezi.pretty_floats(results)))


def logging_results(results, names, tag=''):  \
    logging.info('\t'.join(
      [tag] + ['%s:[%.4f]'%(name, result) for name, result in zip(names, results)]))


def parse_results(results, names=None):
  if type(results[0]) is str:
    temp = results
    results = names
    names = temp
  #only single values in results!
  if names is None:
    return gezi.pretty_floats(results)
  else:
    if len(names) == len(results) - 1:
      names.insert(0, 'loss')
    # if len(names) == 1 and names[0] == 'loss':
    #   names[0] = ''
    if len(names) == len(results):
      return ','.join(gezi.get_value_name_list(results, names))
    else:
      return ','.join(gezi.pretty_floats(results))


def value_name_list_str(results, names=None):
  if names is None:
    return gezi.pretty_floats(results)
  else:
    return gezi.get_value_name_list(results, names)


#-------model load
#def get_model_path(model_dir, model_name=None):
#  """
#  if model_dir ok return latest model in this dir
#  else return orginal model_dir as a model path
#  NOTICE not check if this model path is ok(assume it to be ok)
#  """
#  model_path = model_dir
#  ckpt = tf.train.get_checkpoint_state(model_dir)
#  if ckpt and ckpt.model_checkpoint_path:
#    #@NOTICE below code will be ok int tf 0.10 but fail int 0.11.rc tensorflow ValueError: Restore called with invalid save path
#    #do not use  ckpt.model_checkpoint_path for we might copy the model to other path so the absolute path(you might train using absoluate path) will not match
#    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path))
#    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
#  else:
#    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
#  #assert os.path.exists(model_path), model_path
#  #tf.logging.log_if(tf.logging.WARN, '%s not exist'%model_path, not os.path.exists(model_path))
#  #if not os.path.exists(model_path):
#    #model_path = None
#    #tf.logging.WARN('%s not exist'%model_path)
#    #raise FileNotFoundError(model_path)
#    #raise ValueError(model_path)
#  return model_path


def latest_checkpoint(model_dir, torch=False):
  try:
    if torch or FLAGS.torch:
      files = glob.glob(f'{model_dir}/*.tar') + glob.glob(f'{model_dir}/*.pt')
      files = sorted(files, key=lambda x: os.path.getmtime(x))
      if files:
        filename = open(f'{model_dir}/checkpoint.txt').readline().strip()
        assert filename == os.path.basename(
            files[-1]), f'{filename} {os.path.basename(files[-1])}'
        return files[-1]
      else:
        return None
  except Exception:
    pass
  model_path = get_model_path(model_dir)
  if not model_path:
    files = glob.glob(f'{model_dir}/model.ckpt*')
    files = sorted(files, key=lambda x: os.path.getmtime(x))
    if files:
      model_path = '.'.join(files[-1].split('.')[:-1])
  return model_path


def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path))
    model_path = os.path.join(model_dir,
                              os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(
        model_dir, model_name)
  #if not os.path.exists(model_path+'.index'):
  #  raise ValueError(model_path)
  return gezi.dirname(model_path), model_path


def get_model_dir(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path))
    model_path = os.path.join(model_dir,
                              os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(
        model_dir, model_name)
  #if not os.path.exists(model_path+'.index'):
  #  raise ValueError(model_path)
  return gezi.dirname(model_path)


def get_model_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path))
    model_path = os.path.join(model_dir,
                              os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = None if model_name is None else os.path.join(
        model_dir, model_name)
  #if not os.path.exists(model_path+'.index'):
  #  raise ValueError(model_path)
  return model_path


#cat checkpoint
#model_checkpoint_path: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-256000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-252000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-253000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-254000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-255000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-256000"
def recent_checkpoint(model_dir, latest=False):
  index = -1 if latest else 1
  return open('%s/checkpoint' %
              (model_dir)).readlines()[index].split()[-1].strip('"')


def checkpoint_exists_in(model_dir):
  if not os.path.exists(model_dir):
    return False
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #input valid dir and return latest model
    return True
  elif os.path.isdir(model_dir):
    return False
  else:
    #this might be user specified model like ./model/model-100.ckpt
    #the file exists and we NOTICE we do not check if it is valid model file!
    return True


def get_model_step(model_path):
  return int(model_path.split('/')[-1].split('-')[-1])


def get_model_epoch(model_path):
  try:
    return float(model_path.split('/')[-1].split('-')[-2])
  except Exception:
    return None


def get_model_epoch_from_dir(model_dir):
  model_path = get_model_path(model_dir)
  try:
    return float(model_path.split('/')[-1].split('-')[-2])
  except Exception:
    return None


def get_model_step_from_dir(model_dir):
  model_path = get_model_path(model_dir)
  return int(model_path.split('/')[-1].split('-')[-1])


def save_checkpoint(sess, model_dir, step):
  checkpoint_path = os.path.join(model_dir, 'model.ckpt')
  tf.train.Saver().save(sess, checkpoint_path, global_step=step)


def save_model(model, path, fp16=False):
  tf.keras.mixed_precision.experimental.set_policy('float32')
  gezi.try_mkdir(os.path.dirname(path))
  try:
    model = model.get_model()
  except Exception:
    pass
  try:
    info = gezi.get('info')
    optimizer = info['optimizer']
    optimizer._set_hyper('learning_rate', 0.001)
    model.compile(optimizer=optimizer,
                  loss=info['loss_fn'],
                  metrics=info['metrics'])
  except Exception:
    pass
  if os.path.isdir(path):
    path = os.path.join(path, 'model.h5')

  try:
    if not fp16:
      model.save(path)
    else:
      import pickle
      net = model.to_json()
      weights = [tf.cast(x, tf.float16) for x in model.weights]
      m = {'net': net, 'weights': weights}
      pickle.dump(m, open(path, 'wb'))
    model_size = gezi.get_size(path)
    logging.info(f'melt.save_model to {path} with model_size: {model_size}M')
    run = gezi.get('wandb_run')
    if run:
      wandb.log({'Others/model_size': model_size})
  except Exception as e:
    logging.warning(f'melt.save_model to {path} fail')
    logging.warning(e)


def load_model(path, custom_objects={}, fp16=False):
  custom_objects['tf'] = tf
  if not fp16:
    model = tf.keras.models.load_model(path,
                                       custom_objects=custom_objects,
                                       compile=False)
  else:
    import pickle
    m = pickle.load(open(path, 'rb'))
    model = tf.keras.models.model_from_json(m['net'],
                                            custom_objects=custom_objects)
    model.set_weights(m['weights'])
  try:
    info = gezi.get('info')
    model.compile(optimizer=info['optimizer'],
                  loss=info['loss_fn'],
                  metrics=info['metrics'])
  except Exception:
    pass
  return model


def restore(sess, model_dir, var_list=None, model_name=None):
  assert model_dir
  if var_list is None:
    varnames_in_checkpoint = melt.get_checkpoint_varnames(model_dir)
    #logging.info('varnames_in_checkpoint: {}'.format(varnames_in_checkpoint))
    var_list = slim.get_variables_to_restore(include=varnames_in_checkpoint)
  saver = tf.train.Saver(var_list)
  model_path = get_model_path(model_dir, model_name)
  #assert model_path and os.path.exists(model_path), model_path
  saver.restore(sess, model_path)
  #@TODO still write to file ? using >
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.local_variables_initializer())
  return saver


def restore_from_path(sess, model_path, var_list=None):
  if var_list is None:
    varnames_in_checkpoint = melt.get_checkpoint_varnames(model_path)
    #logging.info('varnames_in_checkpoint: {}'.format(varnames_in_checkpoint))
    var_list = slim.get_variables_to_restore(include=varnames_in_checkpoint)
  saver = tf.train.Saver(var_list)
  saver.restore(sess, model_path)
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.local_variables_initializer())
  return saver


def restore_scope_from_path(sess, model_path, scope):
  variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  saver = tf.train.Saver(variables)
  saver.restore(sess, model_path)
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.local_variables_initializer())
  return saver


def load(model_dir, model_name=None):
  """
  create sess and load from model,
  return sess
  use load for predictor, be sure to build all predict 
  related graph ready before calling melt.load
  """
  sess = get_session()
  restore(sess, model_dir, model_name)
  return sess


def load_from_path(model_path):
  """
  create sess and load from model,
  return sess
  use load for predictor, be sure to build all predict 
  related graph ready before calling melt.load
  """
  sess = get_session()
  restore_from_path(sess, model_path)
  return sess


def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  # TODO check index replace meta
  files = [
      file for file in glob.glob('%s/model.ckpt-*' % (model_dir))
      if not file.endswith('.index')
  ]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files


def variables_with_scope(scope):
  #scope is a top scope here, otherwise change startswith part
  return [v for v in tf.all_variables() if v.name.startswith(scope)]


#@TODO better
def npdtype2tfdtype(dtype, large=False):
  if dtype == np.float32:
    return tf.float32
  if dtype == np.int32:
    if not large:
      return tf.int32
    else:
      return tf.int64
  if dtype == np.int64:
    return tf.int64
  if dtype == np.float64:
    return tf.float32
  return tf.string


def example2inputs(example, keys=None, exclude_keys=[]):
  keys = keys or example.keys()
  keys = [key for key in keys if not key in exclude_keys]
  inputs = {}
  for key in keys:
    try:
      dtype = npdtype2tfdtype(example[key].dtype)
      # if dtype == tf.int64:
      #   dtype = tf.int32
      inputs[key] = tf.keras.layers.Input(shape=example[key].shape,
                                          dtype=dtype,
                                          name=key)
    except Exception:
      logging.debug(f'{key} {example[key]}')
  return inputs


# 注意需要是batch parse
def features2inputs(features, keys=None, exclude_keys=[]):
  keys = keys or features.keys()
  keys = [key for key in keys if not key in exclude_keys and features]
  inputs = {}
  for key in keys:
    try:
      dtype = features[key].dtype
      # if dtype == tf.int64:
      #   dtype = tf.int32
      inputs[key] = tf.keras.layers.Input(shape=features[key].shape[1:],
                                          dtype=dtype,
                                          name=key)
      # inputs[key] = tf.keras.layers.Input(shape=features[key].shape, dtype=features[key].dtype, name=key)
    except Exception:
      logging.debug(f'{key} {features[key]}')
  return inputs


def get_keras_inputs():
  example = gezi.get('dataset_example')
  keys = gezi.get('dataset_features').keys()
  return example2inputs(example, keys)


def load_constant(data_npy,
                  sess=None,
                  trainable=False,
                  dtype=None,
                  shape=None,
                  name=None):
  """
  tf.constant only can be used for small data
  so melt.constant means melt.large_constant and have more general usage
  https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
  """
  name = name or 'constant_data'

  if not hasattr(load_constant, 'constants'):
    load_constant.constants = {}

  if name in load_constant.constants:
    return load_constant.constants[name]

  #or if isinstance(data_npy, str)
  if type(data_npy) is str:
    timer = gezi.Timer('np load %s' % data_npy)
    data_npy = np.load(data_npy)
    timer.print_elapsed()

  if dtype is None:
    dtype = npdtype2tfdtype(data_npy)
  #dtype = tf.float32

  if shape is None:
    shape = data_npy.shape

  # BELOW is ok but since not add to collections in tf_train_flow will not save.., if add to collections=[tf.GraphKeys.GLOBAL_VARIABLES] then sess.run(init_op) still need to feed
  # data_init = tf.placeholder(dtype, shape)
  # #data = tf.get_variable(name=name, dtype=dtype, initializer=data_init, trainable=trainable, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
  # data = tf.get_variable(name=name, dtype=dtype, initializer=data_init, trainable=trainable, collections=[])
  # load_constant.constants[name] = data

  # if sess is None:
  #   sess = melt.get_session()
  # timer = gezi.Timer('sess run initializer')
  # sess.run(data.initializer, feed_dict={data_init: data_npy})
  # timer.print_elapsed()
  # return data

  # TODO below is slow strage, some times not slow.., but should use below and above is just a ungly workaround.. and it has problem not save emb.. so just use below...
  # NOTICE in tf_train_flow sess.run(init_op) will run this again, slow again! TODO better handel
  timer = gezi.Timer('constant_initializer')
  data = tf.get_variable(name,
                         shape=shape,
                         initializer=tf.constant_initializer(data_npy),
                         trainable=trainable)
  load_constant.constants[name] = data
  timer.print_elapsed()

  return data


def load_constant_cpu(data_npy,
                      sess=None,
                      trainable=False,
                      dtype=None,
                      shape=None,
                      name=None):
  with tf.device('/CPU:0'):
    return load_constant(data_npy,
                         sess=sess,
                         trainable=trainable,
                         dtype=dtype,
                         shape=shape,
                         name=name)


def reuse_variables():
  tf.get_variable_scope().reuse_variables()


#---now work.. can set attribute reuse
#def unreuse_variables():
#  tf.get_variable_scope().reuse=None


#------------------------------------tf record save @TODO move to tfrecords
def int_feature(value):
  if not isinstance(value, (list, tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def int64_feature(value):
  if not isinstance(value, (list, tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  if not isinstance(value, (list, tuple)):
    value = [value]
  if not six.PY2:
    if isinstance(value[0], str):
      value = [x.encode() for x in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  if not isinstance(value, (list, tuple)):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


features = lambda d: tf.train.Features(feature=d)

# Helpers for creating SequenceExample objects  copy from \tensorflow\python\kernel_tests\parsing_ops_test.py
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)


def int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def float_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[float_feature(v) for v in values])


def decode_example(x):
  if tf.executing_eagerly():
    x = x.numpy()
  x = tf.train.Example.FromString(x).features.feature
  features = {}
  for key in x.keys():
    typenames = ['bytes_list', 'float_list', 'int64_list']
    dtypes = [np.object, np.float32, np.int64]
    for typename, dtype in zip(typenames, dtypes):
      value = getattr(x[key], typename).value
      if value:
        features[key] = np.array(value, dtype=dtype)
  return features


def first_example(record_file):
  if isinstance(record_file, (list, tuple)):
    record_file = record_file[0]
  if tf.executing_eagerly():
    for item in tf.data.TFRecordDataset(record_file):
      x = decode_example(item)
      return x
  else:
    for item in tf.compat.v1.python_io.tf_record_iterator(record_file):
      x = decode_example(item)
      return x


def init_model(model, Dataset):
  inputs = gezi.list_files(FLAGS.train_input)
  try:
    example = next(
        iter(Dataset('train').make_batch(FLAGS.batch_size, [inputs[0]])))[0]
    model(example)
  except Exception:
    pass


def first_input(record_file):
  example = first_example(record_file)
  for key in example:
    example[key] = np.asarray([example[key]])
  return example


def gen_feature(l, dtype=None):
  if dtype is None:
    if isinstance(l, (str, bytes)):
      dtype = np.str_
    elif isinstance(l, int):
      dtype = np.int64
    elif isinstance(l, float):
      dtype = np.float32
    else:
      dtype = np.asarray(l).dtype
 
  if isinstance(l, Iterable) and dtype != np.str_ and dtype != np.object:
    l = list(l)

  if dtype == np.object or dtype == np.str_:
    try:
      if l.startswith('(') and l.endswith(')'):
        try:
          l = l[1:-1].split(',')
          l = [int(x.strip()) for x in l]
          dtype = np.int64
        except Exception:
          pass
    except Exception:
      pass
  if dtype == np.int64 or dtype == np.int32:
    return melt.int64_feature(l)
  elif dtype == np.float32 or dtype == np.float64:
    return melt.float_feature(l)
  elif dtype == np.object or dtype == np.str_ or dtype.str.startswith('<U'):
    return melt.bytes_feature(l)
  else:
    return melt.bytes_feature(l)


def gen_features(feature, default_value=0):
  feature_ = {}
  for key in feature:
    feature_[key] = feature[key]
    if isinstance(feature[key], list or tuple) and not feature[key]:
      feature_[key] = [default_value]
  for key in feature_:
    try:
      feature_[key] = gen_feature(feature_[key])
    except Exception as e:
      ic('bad key', key, feature_[key])
      ic(e)
      raise (e)
  return feature_


def get_num_records_single(tf_record_file, recount=False):
  if not recount:
    filename = os.path.basename(tf_record_file)
    filename = filename.replace('-', '.').replace('_', '.')
    l = filename.split('.')

    for item in reversed(l):
      if item.isdigit():
        return int(item)

  # try:
  return sum(
      1 for _ in tf.compat.v1.python_io.tf_record_iterator(tf_record_file))
  # except Exception:
  #   return 0


def get_num_records(files, recount=False):
  if isinstance(files, str):
    files = gezi.list_files(files)
  res = sum([
      get_num_records_single(file, recount=recount)
      for file in tqdm(files, ascii=False, desc='get_num_records', leave=False)
  ])
  return res


def get_num_records_print(files):
  num_records = 0
  if isinstance(files, str):
    files = gezi.list_files(files)
  num_inputs = len(files)
  index = 0
  for file in files:
    count = get_num_records_single(file)
    print(file, count, '%.3f' % (index / num_inputs))
    num_records += count
    index += 1
  print('num_records:', num_records)
  return num_records


def load_num_records(input):
  num_records_file = os.path.dirname(input) + '/num_records.txt'
  num_records = int(
      open(num_records_file).read()) if gezi.non_empty(num_records_file) else 0
  return num_records


@gezi.set_timeout(2, lambda: -1)
def get_num_records_from_dir(dir_, recount=False):
  num_records_file = dir_ + '/num_records.txt'
  num_records = -1
  if not recount:
    try:
      num_records = int(open(num_records_file).read()) if gezi.non_empty(
          num_records_file) else -1
      ic('num_records from ', num_records_file, num_records)
    except Exception:
      # logging.warning(traceback.format_exc())
      pass
  if num_records == -1:
    files = gezi.list_files(dir_)
    num_records = get_num_records(files)
    # gezi.write_txt(num_records, num_records_file)
  return num_records


#-------------histogram util
def monitor_train_vars(collections=None):
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var, collections=collections)


class MonitorKeys():
  TRAIN = 'train_monitor'


#@FIXME seems not work get_collection always None
from tensorflow.python.framework import ops


def monitor_gradients_from_loss(loss, collections=[MonitorKeys.TRAIN]):
  grads = tf.gradients(loss, tf.trainable_variables())
  for grad in grads:
    if grad is not None:
      tf.histogram_summary(grad.op.name, grad, collections=collections)
    else:
      raise ValueError('None grad')


#TODO check op.name or .name ? diff?
def histogram_summary(tensor, name=''):
  tf.summary.histogram('{}'.format(name or tensor.op.name), tensor)


def scalar_summary(tensor, name=''):
  tf.summary.scalar('{}'.format(name or tensor.op.name), tensor)


def monitor_embedding(emb, vocab, vocab_size):
  try:
    histogram_summary('emb_0', tf.gather(emb, 0))
    histogram_summary('emb_1', tf.gather(emb, 1))
    histogram_summary('emb_2', tf.gather(emb, 2))
    histogram_summary('emb_1/4', tf.gather(emb, vocab_size // 4))
    histogram_summary('emb_middle', tf.gather(emb, vocab_size // 2))
    histogram_summary('emb_3/4', tf.gather(emb, vocab_size // 4 * 3))
    histogram_summary('emb_end', tf.gather(emb, vocab_size - 1))
    histogram_summary('emb_end2', tf.gather(emb, vocab_size - 2))
    histogram_summary('emb_start_id', tf.gather(emb, vocab.start_id()))
    histogram_summary('emb_end_id', tf.gather(emb, vocab.end_id()))
    histogram_summary('emb_unk_id', tf.gather(emb, vocab.unk_id()))
  except Exception:
    print('monitor_embedding fail', file=sys.stderr)

def visualize_embedding(emb, vocab_txt='vocab.txt'):
  # You can add multiple embeddings. Here we add only one.
  embedding = melt.flow.projector_config.embeddings.add()
  embedding.tensor_name = emb.name
  # Link this tensor to its metadata file (e.g. labels).
  if not vocab_txt.endswith('.project'):
    if vocab_txt.endswith('.bin'):
      embedding.metadata_path = vocab_txt.replace('.bin', '.project')
    elif vocab_txt.endswith('.txt'):
      embedding.metadata_path = vocab_txt.replace('.txt', '.project')
    else:
      embedding.metadata_path = vocab_txt[:vocab_txt.rindex('.')] + '.project'

def get_summary_ops():
  return ops.get_collection(ops.GraphKeys.SUMMARIES)

def print_summary_ops():
  sops = ops.get_collection(ops.GraphKeys.SUMMARIES)
  logging.debug('summary_ops:', sops)

def get_summary_writer(set_walltime=True):
  # if FLAGS.async_eval:
  summary_dir = os.path.join(
      FLAGS.log_dir, 'main') if FLAGS.train_valid_summary else FLAGS.log_dir
  summary_writer = SummaryWriter(summary_dir, set_walltime=set_walltime)
  return summary_writer
  # else:
  #   summary_writer = gezi.get_global('summary_writer', None)
  #   if not summary_writer:
  #     summary_dir = os.path.join(FLAGS.log_dir, 'main') if FLAGS.train_valid_summary else FLAGS.log_dir
  #     summary_writer = SummaryWriter(summary_dir, set_walltime=True)
  #     gezi.set_global('summary_writer', summary_writer)
  #   return summary_writer


def print_global_varaiables(sope=None):
  for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(item)


def print_varaiables(key, sope=None):
  for item in tf.get_collection(key):
    print(item)


def get_global_int(key, val=0):
  if key not in os.environ:
    return val
  return int(os.environ[key])


def get_global_float(key, val=0.):
  if key not in os.environ:
    return val
  return float(os.environ[key])


def get_global_str(key):
  if key not in os.environ:
    os.environ[key] = ''
  return os.environ[key]


def get_global(key, val=None):
  return gezi.get_global(key, val)


def set_global(key, val=None):
  return gezi.set_global(key, val)


def add_global(key, val=None):
  return gezi.add_global(key, val)


def step():
  return get_global('step', 0.)


def epoch():
  return float(get_global('epoch', 0.))


def batch_size():
  return get_global('batch_size')


def eval_batch_size():
  return get_global('eval_batch_size')


def global_batch_size():
  return batch_size()


def global_eval_batch_size():
  return eval_batch_size()


def replica_batch_size():
  return get_global('replica_batch_size')


def replica_eval_batch_size():
  return get_global('replica_eval_batch_size')


def num_gpus():
  return get_global('num_gpus', 1)


def num_gpus2():
  return get_global('num_gpus2', 1)


def loss():
  loss_ = get_global('eval_loss')
  if not loss_:
    loss_ = get_global('train_loss')
  if not loss_:
    loss_ = get_global('loss')
  return loss_


def train_loss():
  return get_global('train_loss')


def eval_loss():
  return get_global('eval_loss')


def duration():
  return get_global('duration')


feed_dict = {}
valid_feed_dict = {}
eval_feed_dict = {}
test_feed_dict = {}


#---------for flow
def default_names(length):
  names = ['metric%d' % (i - 1) for i in range(length)]
  names[0] = 'loss'
  return names


# TODO better handle, just use op.name , but right now has some problem
# especially optimizer will change op.name ... not readable so now
# you have to pass the name by yourself manully
def adjust_names(ops, names):
  assert ops
  if names is None:
    #return [x.name.split('/')[-1].split(':')[0] for x in ops]
    #return [x.name for x in ops]
    return default_names(len(ops))
  else:
    if len(names) == len(ops):
      return names
    elif len(names) + 1 == len(ops):
      names.insert(0, 'loss')
      return names
    elif len(names) + 2 == len(ops):
      names.insert(0, 'loss')
      names.insert(1, 'lr')
    else:
      #return [x.name.split('/')[-1].split(':')[0]for x in ops]
      #return [x.name for x in ops]
      return default_names(len(ops))


def add_summarys(summary, values, names, suffix='', prefix=''):
  for value, name in zip(values, names):
    if suffix:
      summary.value.add(tag='%s/%s' % (name, suffix), simple_value=float(value))
    else:
      if prefix:
        summary.value.add(tag='%s/%s' % (prefix, name),
                          simple_value=float(value))
      else:
        summary.value.add(tag=name, simple_value=float(value))


def add_summary(summary, value, name, suffix='', prefix=''):
  if suffix:
    summary.value.add(tag='%s/%s' % (name, suffix), simple_value=float(value))
  else:
    if prefix:
      summary.value.add(tag='%s/%s' % (prefix, name), simple_value=float(value))
    else:
      summary.value.add(tag=name, simple_value=float(value))


#-----------deal with text  TODO move
# TODO pad for weights start end only zero right now!
import melt


# TODO
def pad_weights(text, weights, start_id=None, end_id=None, end_weight=1.0):
  pass


# TODO simplify without weights
def pad(text, start_id=None, end_id=None, weights=None, end_weight=1.0):
  logging.info('Pad with start_id', start_id, ' end_id', end_id)
  need_start_mark = start_id is not None
  need_end_mark = end_id is not None
  if not need_start_mark and not need_end_mark:
    return text, melt.length(text), weights

  batch_size = tf.shape(text)[0]
  zero_pad = tf.zeros([batch_size, 1], dtype=text.dtype)

  sequence_length = melt.length(text)

  if not need_start_mark:
    text = tf.concat([text, zero_pad], 1)
    if weights is not None:
      weights = tf.concat(
          [weights,
           tf.ones_like(zero_pad, dtype=tf.float32) * end_weight], 1)
  else:
    if need_start_mark:
      start_pad = zero_pad + start_id
      if need_end_mark:
        text = tf.concat([start_pad, text, zero_pad], 1)
        if weights is not None:
          weights = tf.concat([
              tf.zeros_like(start_pad, dtype=tf.float32), weights,
              tf.ones_like(zero_pad, dtype=tf.float32) * end_weight
          ], 1)
      else:
        text = tf.concat([start_pad, text], 1)
        if weights is not None:
          weights = tf.concat(
              [tf.zeros_like(start_pad, dtype=tf.float32), weights], 1)
      sequence_length += 1

  if need_end_mark:
    text = melt.dynamic_append_with_length(
        text, sequence_length, tf.constant(end_id, dtype=text.dtype))
    if weights is not None:
      weights = melt.dynamic_append_with_length_float32(
          weights, sequence_length, tf.constant(end_weight,
                                                dtype=weights.dtype))
    sequence_length += 1

  return text, sequence_length, weights


class GpuHanler(object):

  def __init__(self, num_gpus=None):
    self._cur_gpu = 0

  def next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    if self._num_gpus > 1:
      self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus - 1)
    return dev


def count_records(files):
  import multiprocessing
  from multiprocessing import Value

  counter = Value('i', 0)

  def deal_file(file):
    try:
      count = melt.get_num_records_single(file)
    except Exception:
      print('bad file:', file)
    global counter
    with counter.get_lock():
      counter.value += count

  pool = multiprocessing.Pool()
  pool.map(deal_file, files)
  pool.close()
  pool.join()

  return counter.value


def sparse2dense(features, key=None, default_value=0):

  def sparse2dense_(features, key, default_value):
    val = features[key]
    if val.values.dtype == tf.string:
      default_value = None
    val = melt.sparse_tensor_to_dense(val, default_value)
    features[key] = val

  modified = False
  if key:
    sparse2dense_(features, key)
    modified = True
  else:
    from tensorflow.python.framework.sparse_tensor import SparseTensor
    for key, val in features.items():
      if isinstance(val, SparseTensor):
        sparse2dense_(features, key, default_value)
        modified = True
  return modified


def append_dim(features, keys):
  for key in keys:
    features[key] = tf.expand_dims(features[key], axis=-1)


def try_append_dim(features, keys=None, exclude_keys=[], min_dim=None):
  keys = keys or features.keys()
  keys = [key for key in keys if key not in exclude_keys]
  min_dim = min_dim or int(FLAGS.batch_parse)
  for key in keys:
    # if melt.get_dims(features[key]) == min_dim and features[key].dtype != tf.string:
    if melt.get_dims(features[key]) == min_dim:
      features[key] = tf.expand_dims(features[key], axis=-1)
  return features


try_expand_dim = try_append_dim


def try_remove_dim(features, keys=None, exclude_keys=[]):
  if not isinstance(features, dict):
    return features
  keys = keys or features.keys()
  keys = [key for key in keys if key not in exclude_keys]
  for key in keys:
    try:
      last_dim = melt.get_shape(features[key], -1)
      if melt.get_dims(features[key]) == 2 and isinstance(
          last_dim, int) and last_dim == 1:
        features[key] = tf.squeeze(features[key], axis=-1)
        # print(key, features[key])
    except Exception:
      pass
  return features


try_squeeze_dim = try_remove_dim
try_squeeze = try_squeeze_dim


def hack_tpu_input(features):
  if gezi.get('tpu'):
    keys = list(features.keys())
    for key in keys:
      if features[key].dtype == tf.string:
        del features[key]


class GlobalStep():

  def __init__(self, step):
    self.step = step

  def assign(self, step):
    self.step = step

  def assign_add(self, step):
    self.step += step

  def numpy(self):
    return self.step

  def value(self):
    return self.step


class LearningRate():

  def __init__(self, lr):
    self.lr = lr

  def assign(self, lr):
    self.lr = lr

  def numpy(self):
    return self.lr

  def value(self):
    return self.lr

  def __mul__(self, scalar):
    return self.lr * scalar


def print_model(model, depth=1, print_fn=None, prefix='', cur_depth=1):
  if hasattr(model, 'print_') and model.print_ == False:
    return

  if hasattr(model, 'layers'):
    if not depth or cur_depth < depth:
      for layer in model.layers:
        print_model(layer, depth, print_fn, prefix + '#', cur_depth + 1)

  if not depth or cur_depth <= depth:
    try:
      model.summary(print_fn=print_fn)
    except Exception:
      if depth == 1:
        logging.debug(traceback.format_exc())
      pass


#  total_params = sess.run(tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()]))
#   l2 = sess.run(tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])) / total_params
def get_l2_sum(model):
  if not FLAGS.torch:
    if tf.executing_eagerly():
      l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                    ]).numpy()
    else:
      l2 = melt.get_session().run(
          tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]))
  else:
    l2_fn = torch.nn.MSELoss(reduction='sum')
    l2 = sum(
        l2_fn(p, torch.zeros_like(p)).item()
        for p in model.parameters()
        if p.requires_grad) / 2.
  return l2


def l2_info(model, pre=''):
  total_params = model.count_params()
  l2 = get_l2_sum(model) / total_params
  logging.info(f'{pre} total params: {total_params}, l2:{l2:.5f}')


# https://github.com/keras-team/keras/issues/2717
import tempfile


def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
  # Add L1L2 regularization to the whole model.
  # NOTE: This will save and reload the model. Do not call this function inplace but with
  # model = add_l1l2_regularizer(model, ...)

  if not reg_attributes:
    reg_attributes = [
        'kernel_regularizer', 'bias_regularizer', 'beta_regularizer',
        'gamma_regularizer'
    ]
  if isinstance(reg_attributes, str):
    reg_attributes = [reg_attributes]

  regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)

  for layer in model.layers:
    for attr in reg_attributes:
      if hasattr(layer, attr):
        setattr(layer, attr, regularizer)

  # # So far, the regularizers only exist in the model config. We need to
  # # reload the model so that Keras adds them to each layer's losses.
  # model_json = model.to_json()

  # # Save the weights before reloading the model.
  # tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
  # model.save_weights(tmp_weights_path)

  # # Reload the model
  # model = keras.models.model_from_json(model_json)
  # model.load_weights(tmp_weights_path, by_name=True)

  tmp_model_path = tempfile.gettempdir(), 'tmp_model.h5'
  model.save(tmp_model_path, format='tf')
  model = keras.models.load_model(tmp_model_path)

  return model


def add_weight_decay(model, weight_decay):
  if (weight_decay is None) or (weight_decay == 0.0):
    return

  # recursion inside the model
  def add_decay_loss(m, factor):
    if isinstance(m, tf.keras.Model):
      for layer in m.layers:
        add_decay_loss(layer, factor)
    else:
      for param in m.trainable_weights:
        with tf.keras.backend.name_scope('weight_regularizer'):
          regularizer = lambda: tf.keras.regularizers.l2(factor)(param)
          m.add_loss(regularizer)

  # weight decay and l2 regularization differs by a factor of 2
  add_decay_loss(model, weight_decay / 2.0)
  return


def device(device=None):
  if device == 'gpu':
    device = None
  if device:
    if device == 'cpu':
      device = '/cpu:0'
    return tf.device(device)
  else:
    return gezi.DummyContextManager()


# TODO 有点乱 keras模式目前from_file=True 原来tf1 graph模式默认False
if tf.__version__ > '2':

  def get_eval_step(from_file=True):
    # 如果不是train_loop 模式 eval_round == None
    # from_file=True 表示只从file里获取
    step = 0
    if FLAGS.eval_round is None or from_file:
      step_file = os.path.join(FLAGS.log_dir, 'eval_step.txt')
      if os.path.exists(step_file):
        step = gezi.read_int_from(step_file, 0)
    else:
      step = FLAGS.eval_round
    return step
else:

  def get_eval_step(from_file=False):
    # 如果不是train_loop 模式 eval_round == None
    # from_file=True 表示只从file里获取
    step = 0
    if FLAGS.eval_round is None or from_file:
      step_file = os.path.join(FLAGS.log_dir, 'eval_step.txt')
      if os.path.exists(step_file):
        step = gezi.read_int_from(step_file, 0)
    else:
      step = FLAGS.eval_round
    return step


def inc_eval_step(save_file=False):
  step = 0
  one_step = 1
  if FLAGS.loop_type == 'day':
    one_step = 24 * FLAGS.valid_interval_epochs
    assert one_step == int(one_step)
  if FLAGS.eval_round != None:
    FLAGS.eval_round += one_step
    step = FLAGS.eval_round
  if save_file:
    step_file = os.path.join(FLAGS.log_dir, 'eval_step.txt')
    if not step:
      if os.path.exists(step_file):
        step = gezi.read_int_from(step_file, 0)
      step += 1
    gezi.write_to_txt(step, step_file)
  return step


def save_eval_step(step=None):
  if step is None:
    step = gezi.get('eval_step')
    if not step:
      return
  step_file = os.path.join(FLAGS.log_dir, 'eval_step.txt')
  gezi.write_to_txt(step, step_file)


def inc_train_step():
  step_file = os.path.join(FLAGS.log_dir, 'train_step.txt')
  step = gezi.read_int_from(step_file, 0)
  one_step = 1
  if FLAGS.loop_type == 'day':
    one_step = 24 * FLAGS.valid_interval_epochs
    assert one_step == int(one_step)
  step += one_step
  gezi.write_to_txt(step, step_file)
  return step


def save_train_step(step):
  step_file = os.path.join(FLAGS.log_dir, 'train_step.txt')
  gezi.write_to_txt(step, step_file)


def get_train_step():
  # step_file = os.path.join(FLAGS.log_dir, 'train_step.txt')
  # return gezi.read_int_from(step_file, 0)
  step = 0
  try:
    step = int(
        open(os.path.join(FLAGS.log_dir,
                          'valid_hours.txt')).readlines()[-1].split()[1])
  except Exception:
    pass
  return step


def get_loss_step():
  if not FLAGS.train_loop:
    return 0
  step = gezi.get_global('loss_step', 0)
  if not step:
    step_file = os.path.join(FLAGS.log_dir, 'loss_step.txt')
    if os.path.exists(step_file):
      step = gezi.read_int_from(loss_file, 0)
    gezi.set_global('loss_step', step)
  return step


def inc_loss_step(num_steps=1):
  if not FLAGS.train_loop:
    return 0
  step = get_loss_step()
  step += num_steps
  step_file = os.path.join(FLAGS.log_dir, 'loss_step.txt')
  gezi.write_to_txt(step, step_file)
  gezi.set_global('loss_step', step)
  return step


def get_total_step():
  #if not FLAGS.train_loop:
  #  return 0
  step = gezi.get_global('total_step', 0)
  if not step:
    step_file = os.path.join(FLAGS.log_dir, 'total_step.txt')
    if os.path.exists(step_file):
      step = gezi.read_int_from(step_file, 0)
    gezi.set_global('total_step', step)
  return step


def inc_total_step(num_steps=1):
  #if not FLAGS.train_loop:
  #  return 0
  step = get_total_step()
  step += num_steps
  step_file = os.path.join(FLAGS.log_dir, 'total_step.txt')
  gezi.write_to_txt(step, step_file)
  gezi.set_global('total_step', step)
  return step


def save_total_step(step):
  step_file = os.path.join(FLAGS.log_dir, 'total_step.txt')
  gezi.set_global('total_step', step)
  gezi.write_to_txt(step, step_file)


def set_total_step(step):
  save_total_step(step)
  gezi.set_global('total_step', step)
  return step


def mark_evaluated_model(model):
  gezi.append_to_txt(model, os.path.join(FLAGS.log_dir, 'evaluated_models.txt'))


def is_evaluated_model(model):
  file = os.path.join(FLAGS.log_dir, 'evaluated_models.txt')
  if not os.path.exists(file):
    return False
  for line in open(file):
    line = line.strip()
    if line == model:
      return True
  return False


def mark_evaluating_model(model):
  gezi.append_to_txt(model, os.path.join(FLAGS.log_dir,
                                         'evaluating_models.txt'))


def is_evaluating_model(model):
  file = os.path.join(FLAGS.log_dir, 'evaluating_models.txt')
  if not os.path.exists(file):
    return False
  for line in open(file):
    line = line.strip()
    if line == model:
      return True
  return False


def is_valid_step():
  return FLAGS.valid_interval_steps and gezi.global_step(
  ) % FLAGS.valid_interval_steps == 0


def get_eval_walltime():
  if FLAGS.valid_hour:
    try:
      return datetime.strptime(FLAGS.valid_hour, '%Y%m%d%H').timestamp()
    except Exception:
      return None
  else:
    return None


def write_metric_summaries(names, vals, step, summary=None):
  if FLAGS.write_summary and FLAGS.write_metric_summary:
    if summary is None:
      summary = get_summary_writer()
    logging.debug(f'add summary of step {step} to {FLAGS.log_dir}')
    if vals and names:
      for name, val in zip(names, vals):
        if not isinstance(val, str):
          summary.scalar(name, val, step, walltime=melt.get_eval_walltime())
        else:
          logging.warning('bad val!', name, val)
      summary.scalar('other/rounds', step, step, 0)
      summary.scalar('other/rounds2', step, step)
      if FLAGS.train_hour and FLAGS.valid_hour and len(
          FLAGS.valid_hour) == len('20200101'):
        try:
          summary.scalar('other/valid_span',
                         gezi.diff_hours(FLAGS.valid_hour, FLAGS.train_hour),
                         step)
        except Exception:
          pass


def get_padding_values(features, value=1):
  padding_values = {}
  for key in features:
    if features[key].dtype == tf.float32:
      padding_values[key] = float(value)
    elif features[key].dtype == tf.int64:
      padding_values[key] = value
    else:
      padding_values[key] = ''
  return padding_values


def lookup_feats(x, emb, feat_names, feat_lens):
  feats = emb(x)
  # print('------', feats)
  feats = tf.split(feats, feat_lens, axis=-1)
  res = {}
  for i, name in enumerate(feat_names):
    res[name] = feats[i]
  return res


def tonumpy(x):
  if not isinstance(x, dict):
    return x.numpy()
  else:
    for key in x:
      x[key] = x[key].numpy()
    return x


def batch_len(x):
  if not isinstance(x, dict):
    return len(x)
  else:
    return len(next(iter(x.values())))


def recompile_model(model, **kwargs):
  model.compile(
      loss=model.my_loss,
      optimizer=model.my_optimizer,
      # metrics=model.my_metrics,
      metrics=gezi.get('info')['metrics'],
      **kwargs)
  return model


def set_trainable(model, recompile=False, **kwargs):
  for layer in model.layers:
    layer.trainable = True
  if recompile:
    recompile_model(model, **kwargs)
  return model


# https://github.com/qubvel/segmentation_models/blob/94f624b7029deb463c859efbd92fa26f512b52b8/segmentation_models/models/_utils.py
def freeze_model(model, feeze_all=False, **kwargs):
  for layer in model.layers:
    if not isinstance(layer, layers.BatchNormalization) or freeze_all:
      layer.trainable = False
  return model


def get_model_input(model):
  inputs = model.input
  if isinstance(inputs, dict):
    inputs_ = {}
    for key in inputs:
      inputs_[key] = tf.keras.layers.Input(inputs[key].shape[1:],
                                           dtype=inputs[key].dtype,
                                           name=inputs[key].name)
  elif isinstance(inputs, (list, tuple)):
    inputs_ = [
        tf.keras.layers.Input(input.shape[1:],
                              dtype=input.dtype,
                              name=input.name) for input in inputs
    ]
  else:
    inputs_ = tf.keras.layers.Input(inputs.shape[1:],
                                    dtype=inputs.dtype,
                                    name=inputs.name)
  return inputs_


class EnsembleModel(Model):

  def __init__(self,
               models,
               weights=[],
               activation=None,
               cpu_merge=False,
               **kwargs):
    super(EnsembleModel, self).__init__(**kwargs)
    assert models
    if isinstance(models[0], str):
      models = [load_model(m) for m in tqdm(models, desc='Ensemble loading')]
    self.models = models
    for i in range(len(models)):
      self.models[i]._name = self.models[i].name + '_' + str(i)
    self.models_weights = list(map(float, weights))
    self.activation = tf.keras.activations.get(activation)
    self.cpu_merge = cpu_merge
    self.name_ = f'Ensemble_{len(models)}'

  def call(self, x):
    if not self.cpu_merge:
      xs = [model(x) for model in self.models]
    else:
      xs = []
      for model in self.models:
        res = model(x)
        with tf.device('/cpu:0'):
          res = tf.identity(res)
          xs.append(res)

    device_ = 'gpu' if not self.cpu_merge else 'cpu'
    with device(device_):
      reduce_fn = tf.reduce_mean

      if self.models_weights:
        reduce_fn = tf.reduce_sum
        assert len(self.models_weights) == len(xs)
        xs = [
            self.activation(xs[i]) * self.models_weights[i]
            for i in range(len(xs))
        ]
      else:
        xs = [self.activation(xs[i]) for i in range(len(xs))]

      x = reduce_fn(tf.stack(xs, axis=1), axis=1)

    return x

  def get_model(self):
    try:
      inp = self.models[0].input
      out = self.call(inp)
      return tf.keras.Model(inp, out, name=self.name_)
    except Exception as e:
      print(e)
      return self


# TODO fp16相关单独一个模块儿
def fp16(x):
  dtype = tf.float16 if not gezi.get('tpu') else tf.bfloat16
  return tf.cast(x, dtype=dtype)


def fp16_policy():
  policy_name = 'mixed_float16' if not gezi.get('tpu') else 'mixed_bfloat16'
  return policy_name


def set_fp16_policy():
  # 注意tpu本来能跑的程序设置fp16反而OOM了。。 而且tpu本身内部就是混合精度吧 那mixed_bfloat16是否就没必要了? 不是tpu也需要手动设置bfloat16模式开启 似乎改成 repeat_then_shuffle模式没有问题了 速度也确实变快
  # 但是很难启动成功 就成功了一次 速度从5.6min/epoch提升到 4.2min /epoch 但是大部分情况都无法启动 报错OOM反而 Attempting to reserve 6.15G at the bottom of memory. That was not possible. There are 6.95G free, 0B reserved, and 4.79G reservable
  # fp32没有这个问题
  # gpu fp16测试是有效的 能开更大batch * 2, 而且速度加快 但是注意开大batch有可能会 Error polling for event status: failed to query event: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
  # 出现一些cuda 错误 不知道是否和tf版本 cuda version有关 TODO 把batch size减小会训练稳定
  # 开启fp16混合精度gpu训练和tpu训练的速度差距缩小了
  # fp16 gpu 当前无法save checkpoint
  #  WARNING: A slot variable was re-used as a dependency of a Trackable object. This is not currently allowed. File a feature request if this limitation bothers you.
  # 总结 tf2.3.0测试 gpu fp16 不稳定 无法save checkpoint 可能cuda error
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  policy_name = 'mixed_float16' if not gezi.get('tpu') else 'mixed_bfloat16'
  gezi.set('precision_policy_name', policy_name)
  if policy_name == 'mixed_float16':
    # HACK save 会失败  WARNING: A slot variable was re-used as a dependency of a Trackable object. This is not currently allowed. File a feature request if this limitation bothers you.
    FLAGS.save_checkpoint = False
  policy = mixed_precision.Policy(policy_name)
  mixed_precision.set_policy(policy)


# ------------- https://github.com/google/automl/blob/master/efficientdet/utils.py
import contextlib
from typing import Text, Tuple, Union


def get_precision(strategy: str, mixed_precision: bool = False):
  """Get the precision policy for a given strategy."""
  if mixed_precision:
    if strategy == 'tpu':
      return 'mixed_bfloat16'

    if tf.config.experimental.list_physical_devices('GPU'):
      return 'mixed_float16'

    # TODO(fsx950223): Fix CPU float16 inference
    # https://github.com/google/automl/issues/504
    logging.warning('float16 is not supported for CPU, use float32 instead')
    return 'float32'

  return 'float32'


@contextlib.contextmanager
def float16_scope():
  """Scope class for float16."""

  def _custom_getter(getter, *args, **kwargs):
    """Returns a custom getter that methods must be called under."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.float16:
      kwargs['dtype'] = tf.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    if cast_to_float16:
      var = tf.cast(var, tf.float16)
    return var

  with tf.variable_scope('', custom_getter=_custom_getter) as varscope:
    yield varscope


def set_precision_policy(policy_name: Text = None, loss_scale: bool = True):
  """Set precision policy according to the name.
  Args:
    policy_name: precision policy name, one of 'float32', 'mixed_float16',
      'mixed_bfloat16', or None.
    loss_scale: whether to use loss scale (only for training).
  """
  if not policy_name:
    return

  assert policy_name in ('mixed_float16', 'mixed_bfloat16', 'float32')
  logging.info('use precision policy name %s', policy_name)
  # tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
  # mixed_float16 training is not supported for now, so disable loss_scale.
  # float32 and mixed_bfloat16 do not need loss scale for training.
  if loss_scale:
    policy = tf.keras.mixed_precision.experimental.Policy(policy_name)
  else:
    policy = tf.keras.mixed_precision.experimental.Policy(policy_name,
                                                          loss_scale=None)
  tf.keras.mixed_precision.experimental.set_policy(policy)


def build_model_with_precision(pp, mm, ii, *args, **kwargs):
  """Build model with its inputs/params for a specified precision context.
  This is highly specific to this codebase, and not intended to be general API.
  Advanced users only. DO NOT use it if you don't know what it does.
  NOTE: short argument names are intended to avoid conficts with kwargs.
  Args:
    pp: A string, precision policy name, such as "mixed_float16".
    mm: A function, for rmodel builder.
    ii: A tensor, for model inputs.
    tt: A bool, If true, it is for training; otherwise, it is for eval.
    *args: A list of model arguments.
    **kwargs: A dict, extra model parameters.
  Returns:
    the output of mm model.
  """
  tt = True
  if pp == 'mixed_bfloat16':
    set_precision_policy(pp)
    inputs = tf.cast(ii, tf.bfloat16)
    with tf.tpu.bfloat16_scope():
      outputs = mm(inputs, *args, **kwargs)
    set_precision_policy('float32')
  elif pp == 'mixed_float16':
    set_precision_policy(pp, loss_scale=tt)
    inputs = tf.cast(ii, tf.float16)
    with float16_scope():
      outputs = mm(inputs, *args, **kwargs)
    set_precision_policy('float32')
  elif not pp or pp == 'float32':
    outputs = mm(ii, *args, **kwargs)
  else:
    raise ValueError('Unknow precision name {}'.format(pp))

  # Users are responsible to convert the dtype of all outputs.
  return outputs


def _recompute_grad(f):
  """An eager-compatible version of recompute_grad.
  For f(*args, **kwargs), this supports gradients with respect to args or
  kwargs, but kwargs are currently only supported in eager-mode.
  Note that for keras layer and model objects, this is handled automatically.
  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.
  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  """

  @tf.custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    current_var_scope = tf.compat.v1.get_variable_scope()
    from tensorflow.python.eager import tape as tape_lib
    with tape_lib.stop_recording():
      kwargs_ = {}
      for key in kwargs:
        if key != 'training':
          kwargs_[key] = kwargs[key]
      result = f(*args, **kwargs_)

    def grad_wrapper(*wrapper_args, **grad_kwargs):
      """Wrapper function to accomodate lack of kwargs in graph mode decorator."""

      @tf.custom_gradient
      def inner_recompute_grad(*dresult):
        """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
        # Gradient calculation for reverse mode autodiff.
        variables = grad_kwargs.get('variables')
        with tf.GradientTape() as t:
          id_args = tf.nest.map_structure(tf.identity, args)
          t.watch(id_args)
          if variables is not None:
            t.watch(variables)
          with tf.control_dependencies(dresult):
            with tf.variable_scope(current_var_scope):
              result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
          kw_vars = list(variables)
        grads = t.gradient(result,
                           list(id_args) + kw_vars,
                           output_gradients=dresult,
                           unconnected_gradients=tf.UnconnectedGradients.ZERO)

        def transpose(*t_args, **t_kwargs):
          """Gradient function calculation for forward mode autodiff."""
          # Just throw an error since gradients / activations are not stored on
          # tape for recompute.
          raise NotImplementedError(
              'recompute_grad tried to transpose grad of {}. '
              'Consider not using recompute_grad in forward mode'
              'autodiff'.format(f.__name__))

        return (grads[:len(id_args)], grads[len(id_args):]), transpose

      return inner_recompute_grad(*wrapper_args)

    return result, grad_wrapper

  return inner


def recompute_grad(recompute=False):
  """Decorator determine whether use gradient checkpoint."""

  def _wrapper(f):
    if recompute:
      return _recompute_grad(f)
    return f

  return _wrapper


##---------------------------
# modify from https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/utils/vis_utils.py#L259-L325
# 注意expand_nested设置True 可能会报错失败, 默认dpi 96 -> 360, shwo_shapes置为True
def plot_model(model,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=360):
  """Converts a Keras model to dot format and save to a file.
  Example:
  ```python
  input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
  x = tf.keras.layers.Embedding(
      output_dim=512, input_dim=10000, input_length=100)(input)
  x = tf.keras.layers.LSTM(32)(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
  model = tf.keras.Model(inputs=[input], outputs=[output])
  dot_img_file = '/tmp/model_1.png'
  tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
  ```
  Arguments:
    model: A Keras model instance
    to_file: File name of the plot image.
    show_shapes: whether to display shape information.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot:
        'TB' creates a vertical plot;
        'LR' creates a horizontal plot.
    expand_nested: Whether to expand nested models into clusters.
    dpi: Dots per inch.
  Returns:
    A Jupyter notebook Image object if Jupyter is installed.
    This enables in-line display of the model plots in notebooks.
  """
  dot = tf.keras.utils.model_to_dot(model,
                                    show_shapes=show_shapes,
                                    show_layer_names=show_layer_names,
                                    rankdir=rankdir,
                                    expand_nested=expand_nested,
                                    dpi=dpi)
  # to_file = path_to_string(to_file)
  if dot is None:
    return
  _, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  # Save image to disk.
  dot.write(to_file, format=extension)
  if extension != 'pdf':
    from PIL import Image
    image = np.asarray(Image.open(to_file))
  else:
    # TODO Unable to get page count. Is poppler installed and in PATH
    from pdf2image import convert_from_path
    images = convert_from_path(to_file)
    image = np.asarray(images[0])

  return image


def plot_model_notebook(model,
                        to_file='model.png',
                        show_shapes=True,
                        show_layer_names=True,
                        rankdir='TB',
                        expand_nested=False,
                        dpi=360):
  return tf.keras.utils.plot_model(model, to_file, show_shapes,
                                   show_layer_names, rankdir, expand_nested,
                                   dpi)


def model2tb(model):
  if not gezi.get(f'model2tb_{model.name}'):
    try:
      model_ = model.get_model() if hasattr(model, 'get_model') else model
      # model_graph = plot_model(model_, to_file=f'{FLAGS.log_dir}/graph.pdf', dpi=120)
      model_graph = plot_model(model_,
                               to_file=f'{FLAGS.log_dir}/graph.png',
                               dpi=360)
      if not FLAGS.wandb or FLAGS.wandb_tb:
        logger = gezi.get_summary_writer()
        logger.image('Model', model_graph, 0)
      else:
        import wandb
        wandb.log({'Model': [wandb.Image(model_graph, caption=model_.name)]})
    except Exception as e:
      logging.debug(f'model2tb for model {model.name} fail')
      logging.debug(e)
    gezi.set(f'model2tb_{model.name}', True)


def swish_activation(x):
  return (K.sigmoid(x) * x)


class FixedDropout(tf.keras.layers.Dropout):

  def _get_noise_shape(self, inputs):
    if self.noise_shape is None:
      return self.noise_shape

    symbolic_shape = K.shape(inputs)
    noise_shape = [
        symbolic_shape[axis] if shape is None else shape
        for axis, shape in enumerate(self.noise_shape)
    ]
    return tuple(noise_shape)


def get_custom_objects():
  return {
      'tf': tf,
      'swish_activation': swish_activation,
      'FixedDropout': FixedDropout
  }


def append_model_suffix(suffix):
  if FLAGS.mode == 'async_valid':
    return

  if FLAGS.mn:
    if FLAGS.mn.endswith(suffix):
      return
    if FLAGS.mode != 'train':
      return

  if suffix.startswith('.'):
    FLAGS.mn += suffix
  else:
    FLAGS.mn += f'.{suffix}'

def get_mode():
  return FLAGS.mode or FLAGS.work_mode

def get_float():
  if FLAGS.fp16:
    return tf.float16
  return tf.float32

def save_dense(dense, dir):
  gezi.try_mkdir(dir)
  np.save(f'{dir}/kernel.npy', dense.kernel)
  if hasattr(dense, 'bias'):
    np.save(f'{dir}/bias.npy', dense.bias)

def load_dense(dir, name=None):
  emb = np.load(f'{dir}/kernel.npy')
  if not os.path.exists(f'{dir}/bias.npy'):
    use_bias = False
    bias_initializer = tf.constant_initializer(np.load(f'{dir}/bias.npy'))
    kernel_initializer = tf.constant_initializer(embedding_lookup_mean)
  else:
    use_bias = True
    kernel_initializer = tf.constant_initializer(emb)
    bias_initializer = 'zeros'

  return tf.keras.layers.Dense(emb.shape[-1], 
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer,
    name=name
    )
