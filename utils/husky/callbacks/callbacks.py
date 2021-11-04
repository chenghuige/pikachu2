#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   callbacks.py
#        \author   chenghuige  
#          \date   2019-09-04 19:42:01.317684
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

import sys 
import os
import traceback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras import layers

try:
  import tensorflow_model_optimization as tfmot
except Exception:
  pass

try:
  import wandb
  HAS_WANDB = True
except ImportError:
  HAS_WANDB = False
  pass

import pandas as pd
import numpy as np
from gezi import tqdm
import traceback
import inspect
import math
import timeit
import copy

import melt
from melt.distributed import tonumpy
import gezi 
logging = gezi.logging

from collections import defaultdict
from husky.callbacks.tqdm_progress_bar import TQDMProgressBar

class AccGradientsCallback(Callback):
  def __init__(self):
    self.step = melt.get_total_step()
    self.total_steps = gezi.get('total_steps')

  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    self.model.step = self.step
    self.model.first = False # TODO not work ?
    if self.step >= self.total_steps:
      try:
        self.model.try_apply_accu_gradients()          
      except Exception:
        pass

  def on_train_end(self, logs={}):
    try:
      self.model.try_apply_accu_gradients()          
    except Exception:
      pass

class NBatchLoggerCallback(Callback):
  """
  A Logger that log average performance per `display` steps.
  """
  def __init__(self, display):
    self.step = melt.get_total_step()
    self.display = display
    self.metric_cache = defaultdict(int)
    self.count_cache = defaultdict(int)
    self.logger = None
    if FLAGS.log_dir and FLAGS.write_summary and FLAGS.write_metric_summary:
      self.logger = melt.get_summary_writer()
    self.total_step = melt.get_total_step()

  # 主要收集 train loss
  def on_batch_end(self, batch, logs={}):
    # tf.print('----logs', logs)
    # exit(0)
    if 'loss' in logs:
      gezi.set('loss', logs['loss'])
    if 'val_loss' in logs:
      gezi.set('val_loss', logs['val_loss'])
    gezi.set('logs', logs)
      
    self.step += FLAGS.steps_per_execution
    if not self.display:
      return

    for k in logs:
      # ValidLossCallback have done this
      if k.startswith('val_'):
        continue

      #TODO MIoU 大写字母开头的默认不加train 只加valid 因为当前trian 无法控制metric执行频次
      if k[0].isupper():
        continue

      # those are customed metrics only for eval data, and done logs in EvaluateCallback
      if 'val_' + k not in logs:
        continue

      self.metric_cache[k] += logs[k]
      self.count_cache[k] += 1
    if self.step % self.display == 0:
        for (k, v) in self.metric_cache.items():
          val = v / self.count_cache[k]
          if FLAGS.valid_interval_steps and self.step % FLAGS.valid_interval_steps == 0:
            prefix = 'train' if not 'loss' in k else 'Loss'
            name = k if k != 'loss' else 'train'
            if self.logger:
              self.logger.scalar(f'{prefix}/{name}', val, self.step + self.total_step)
            if k == 'loss':
              gezi.get('history')['loss'].append(val)
              
        self.metric_cache.clear()
        self.count_cache.clear()

# 有可能某些操作TPU兼容有问题 可以设置 FLAGS.valid_interval_steps = 0 屏蔽掉
class ValidLossCallback(Callback):
  def __init__(self, valid_data, loss_fn=None):
    self.valid_data = valid_data
    # strategy = melt.distributed.get_strategy() #not work 'PerReplicaSpec' object has no attribute '_to_batched_tensor_list'
    # self.valid_data = strategy.experimental_distribute_dataset(valid_data) 
    self.valid_iter = iter(self.valid_data) if self.valid_data else None
    self.loss_fn = loss_fn
    self.step = melt.get_total_step()
    self.val_loss = 0.
    if FLAGS.log_dir and FLAGS.write_summary and FLAGS.write_metric_summary:
      self.logger = melt.get_summary_writer()

    self.total_step = melt.get_total_step()
    self.keval_ok = True

  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    if self.valid_data is None:
      return

    if FLAGS.valid_interval_steps:
      if self.step % FLAGS.valid_interval_steps == 0 or self.step in [1, 100, 200]:
        learning_phase = K.learning_phase()
        K.set_learning_phase(0)
        # tf 2.4开始这里有warning.. 
        # tensorflow/core/grappler/optimizers/data/auto_shard.cc:656] In AUTO-mode, and switching to DATA-based sharding, instead of FILE-based sharding as we cannot find appropriate reader dataset op(s) to shard. Error: Did not find a shardable source, walked to a node which is not a dataset: name: "FlatMapDataset/_9"
        #  Consider either turning off auto-sharding or switching the auto_shard_policy to DATA to shard this dataset. You can do this by creating a new `tf.data.Options()` object then setting `options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA` before applying the options object to the dataset via `dataset.with_options(options)`.
        x, y = next(self.valid_iter)
        res = {'loss': np.nan}
        if self.keval_ok:
          try:
            # 应该还是和train共享的Metric对象update 只是loss 每个batch独立计算不累积 如果是auc mean_iou这样的累积型train valid放在一起迭代有问题
            res = self.model.evaluate(x, y, steps=1, verbose=0, return_dict=True)
            # print('----------before valid', self.step, res, K.learning_phase())
            # print('----dropout', layers.Dropout(0.5)(tf.ones((100,), tf.float32)))
            # x, y = next(self.valid_iter)
            # res = self.model.evaluate(x, y, steps=1, verbose=0, return_dict=True)
            # print('----------', self.step, res)
            # x, y = next(self.valid_iter)
            # res = self.model.evaluate(x, y, steps=1, verbose=0, return_dict=True)
            # print('----------', self.step, res)
            # # TODO FIXME test_on_batch 似乎结果不对。。 why ?   sh ./train/v17/base.sh --mn=test --clear_first 复现 valid_loss 看起来和train loss差不多 而因为valid负样本多 valid loss应该比train低很多
            ## 但是连续执行model.evaluate model.test_on_batch 输出一致 所以.. ?  
            # res = self.model.test_on_batch(x, y, return_dict=True)
          except Exception:
            self.keval_ok = False
            self.valid_data = None
            logging.warning(traceback.format_exc())
        
        # if not self.keval_ok:
        #   y_ = self.model(x)
        #   res['loss'] = self.loss_fn(y, y_)
        K.set_learning_phase(learning_phase)

        val_loss = res['loss']
        gezi.set('val_loss', val_loss)
        
        # # HACK
        # metrics = list(res.keys())
        # for metric in metrics:
        #   if metric[0].isupper() or (metric.startswith('val_') and metric[4].isupper()):
        #     del res[metric]

        for key in res:
          logs[f'val_{key}'] = res[key]

        step = self.step + self.total_step
        if self.logger:
          self.logger.scalar('Loss/valid', val_loss, step) 
        gezi.get('history')['val_loss'].append(val_loss)
        for key in res:
          if key != 'loss': 
            gezi.get('history')[f'val_{key}'].append(res[key])
            # print(key, res[key], step)
            if self.logger:
              try:
                self.logger.scalar(f'valid/{key}', res[key], step) 
              except Exception as e:
                logging.error(e)
        self.val_loss = val_loss
      else:
        logs['val_loss'] = self.val_loss

class TFModelSaverCallback(Callback):
  def __init__(self, interval, steps_per_epoch, variables_to_save=None):
    self.saver = tf.compat.v1.train.Saver(
      max_to_keep=FLAGS.max_models_keep, 
      var_list=variables_to_save) 

    self.interval = interval
    self.steps_per_epoch = steps_per_epoch
    self.step = melt.get_total_step()

  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    finish = int(self.step / self.steps_per_epoch) == FLAGS.num_epochs 
    if self.step % self.interval == 0 or finish:
      if not FLAGS.train_hour:
        model_path = os.path.join(FLAGS.model_dir, 'model.ckpt-%.2f'%(self.step / float(self.steps_per_epoch)))
      else:
        model_path = os.path.join(FLAGS.model_dir, 'model.ckpt-%s-%.2f'%(FLAGS.train_hour, self.step / float(self.steps_per_epoch)))
      model_step_path = model_path + '-' + str(self.step)
      sess = melt.get_session()
      self.saver.save(sess, model_step_path)

      if finish:
        if hasattr(self.model, 'init_predict'):
          self.model.init_predict()
        all_keys = sess.graph.get_all_collection_keys()
        exclude_keys = set(['variables', 'queue_runners', 'summaries', 'train_op', 'update_ops', 'model_variables', 'cond_context', 'while_context'])
        output_collection_names = [x for x in all_keys if x not in exclude_keys and not 'train' in x and not x.endswith('_end_points')]
        logging.info('all collection keys: {}'.format(all_keys[:100]))
        logging.info('collection names to freeze: {}'.format(output_collection_names))
        melt.freeze_graph(os.path.join(FLAGS.model_dir, 'model.pb'), output_collection_names)

class CheckpointCallback(Callback):
  def __init__(self, ckpt_dir, model, optimizer=None, iterator=None, pretrained_dir=None, **kwargs):
    # opt = tf.keras.optimizers.Adam(0.1)
    # net = Net()
    # dataset = toy_dataset()
    # iterator = iter(dataset)
    # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
    if iterator is not None:
      kwargs['iterator'] = iterator
    self.ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, **kwargs)
    self.ckpt_dir = ckpt_dir
    self.pretrained_dir = pretrained_dir
    self.optimizer = optimizer
    train_hour = FLAGS.train_hour if FLAGS.train_loop else None 
    # try:
    #   # should be like 20200324 2020032425
    #   x = int(train_hour)
    # except Exception:
    #   train_hour = None
    checkpoint_name = 'model.ckpt' if not train_hour else f'model.ckpt-{train_hour}'
    self.checkpoint_name = checkpoint_name
    self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=1, checkpoint_name=checkpoint_name)
    self.model = model
    self.loaded = None
    self.step = melt.get_total_step()
    self.saved_step = None
    self.logging = logging.debug if FLAGS.steps >= 0 else logging.info

  def load_models(self, ckpt_dir):
    loaded = False
    model_names = FLAGS.model_names
    num_models = len(model_names)
    root = os.path.dirname(ckpt_dir)
    models = []
    for model_name in tqdm(model_names, ascii=True, desc='load_models'):
      path = f'{root}/{model_name}' if not os.path.exists(model_name) else model_name
      models.append(self.load_model(path, return_model=True))
      assert models[-1]

    loaded = True

    # Can't set the attribute "name", likely because it conflicts with an existing read-only @property of the object. Please choose a different name
    for i, model in enumerate(models):
      model._name += f'_{i}'
    weights = list(map(float, FLAGS.model_weights))
    model = melt.EnsembleModel(models, weights, activation=FLAGS.ensemble_activation, cpu_merge=FLAGS.cpu_merge, 
                               name=f'Ensemble.{num_models}.' + '_'.join(model_names))
    try:
      model = model.get_model()
    except Exception as e:
      logging.debug(e)
    self.model = model
    return loaded

  def load_model(self, ckpt_dir, return_model=False):
    loaded = False
    model = self.model
    # https://www.codeleading.com/article/63411651006/
    # https://github.com/keras-team/keras/issues/5298 why need custom_objects={'tf': tf}
    if os.path.exists(f'{ckpt_dir}/saved_model'):
      try:
        model = tf.keras.models.load_model(f'{ckpt_dir}/saved_model', compile=False, custom_objects={'tf': tf})
        logging.info(f'load_model(graph&weights): [{ckpt_dir}/saved_model] with compile=False')
        loaded = True
      except Exception as e:
        logging.debug(e)
    if not loaded and os.path.exists(f'{ckpt_dir}/model.h5'):
      try:
        try:
          # TOOD custom_object = {'x': x} ?
          model = tf.keras.models.load_model(f'{ckpt_dir}/model.h5', compile=False, custom_objects={'tf': tf})
          logging.info(f'load_model(graph&weights): [{ckpt_dir}/model.h5] with compile=False')
          loaded = True
        except Exception as e:
          logging.debug(e)
          # logging.info('load_model with compile=False')
          # model = tf.keras.models.load_model(f'{ckpt_dir}/model.h5', compile=False)
      except Exception as e:
        logging.error(e)

    if loaded:
      if return_model:
        return model
      self.model = model
    else:
      logging.debug(f'Load model(with graph) from {ckpt_dir} fail')
      if return_model:
        return None
    
    return loaded
    
  def load(self, ckpt_dir):
    loaded = False
    
    total_params = 1
    try:
      total_params = self.model.count_params()
      l2 = melt.get_l2_sum(self.model) / total_params
    except Exception as e:
      logging.warning(e)
      l2 = 0.
    logging.info(f'before loading, total params: {total_params}, l2:{l2:.6f}')
    num_layers = len(self.model.layers)
    logging.debug(f'model has {num_layers} layers', [x.name for x in self.model.layers])
    
    if (FLAGS.work_mode != 'train' or FLAGS.steps == -1) and FLAGS.load_graph and (not FLAGS.load_weights_only):
      # 非训练模式尝试直接载入完整graph + weights
      loaded = self.load_model(ckpt_dir) if not FLAGS.model_names else self.load_models(ckpt_dir)
      if loaded:
        logging.info('keras.models.load_model:', loaded, 'ckpt_dir', ckpt_dir)

    if not loaded:
      if not FLAGS.load_weights_only:
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        text = f'latest ckpt to restore: [{latest_ckpt}]'
        if not latest_ckpt:
          text += f' from {ckpt_dir}'
      else:
        latest_ckpt = None
      if latest_ckpt != None:
        try:
          options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
          self.ckpt.restore(latest_ckpt, options=options)
          logging.info(text)
          loaded = True
        except Exception as e:
          logging.debug(e)
      else:
        logging.debug(f'No checkpoint found in {ckpt_dir}')

      if not loaded:
        if os.path.exists(f'{ckpt_dir}/model.h5'):
          try:
            self.model.load_weights(f'{ckpt_dir}/model.h5')
            logging.info(f'load_weights: [{ckpt_dir}/model.h5]')
            loaded = True
          except Exception as e:
            logging.debug(e)
            if FLAGS.load_by_name:
              try:
                logging.warning('load weights by topological order fail, try by_name=True')
                self.model.load_weights(f'{ckpt_dir}/model.h5', by_name=True)
                loaded = True     
              except Exception as e:
                logging.warning(e)
                if FLAGS.load_skip_mismatch:
                  logging.warning('load weights by topological order fail, try by_name=True and skip_mismatch=True')
                  self.model.load_weights(f'{ckpt_dir}/model.h5', by_name=True, skip_mismatch=True)
                  loaded = True     
        elif os.path.exists(f'{ckpt_dir}/model_weights.h5'):
          logging.info(f'load_weights: [{ckpt_dir}/model_weights.h5]')
          self.model.load_weights(f'{ckpt_dir}/model_weights.h5', by_name=FLAGS.load_by_name)
          loaded = True          
        elif os.path.exists(f'{ckpt_dir}/model_weights.bin'):
          logging.info(f'load_weights: [{ckpt_dir}/model_weights.bin]')
          self.model.load_weights(f'{ckpt_dir}/model_weights.bin', by_name=FLAGS.load_by_name)
          loaded = True  
        else:
          logging.info(f'No model.h5 or other weights file found in {ckpt_dir}')       

    if loaded:
      total_params = self.model.count_params()
      l2_ = l2
      try:
        l2 = melt.get_l2_sum(self.model) / total_params
      except Exception as e:
        logging.warning(e)
        l2 = 0.
      logging.info(f'after loading, total params: {total_params}, l2:{l2:.6f}, l2_diff:{l2 - l2_:.6f}')
      # assert abs(l2 - l2_) > K.epsilon(), 'Loading fail, no weights updated'
      if abs(l2 - l2_) < K.epsilon():
        logging.warning('Too small weights change before and after loading, might due to some error that no weights were updated')

    return loaded

  def on_train_begin_(self, logs={}):
    if FLAGS.cv_valid_only:
      return 

    if (FLAGS.round == 0 and self.loaded is None) or  FLAGS.fold is not None:
      loaded = self.load(self.ckpt_dir)
      if FLAGS.check_loaded:
        assert loaded, 'no model loaded!'
        
      if not loaded and self.pretrained_dir:
        ic(self.pretrained_dir)
        loaded = self.load(self.pretrained_dir)
        
      self.loaded = loaded

  def save(self, is_last=False, logs={}):
    num_layers = len(self.model.layers)
    logging.debug(f'save model weith {num_layers} layers', [x.name for x in self.model.layers])

    if is_last:
      swa = gezi.get('swa')
      if swa is not None:
        swa.finalize()

      if hasattr(self.model, 'custom_save'):
        self.model.custom_save()
    
    saved = False
    if FLAGS.save_checkpoint:
      # hack for tpu
      save_checkpoint = False
      if gezi.get('tpu'):
        # now_time = timeit.default_timer()
        # start_time = gezi.get('start_time')
        # # colab can train 24 hours
        # if start_time and now_time - start_time > 22 * (60 * 60):
        #   save_checkpoint = True
        # if is_last:
        #   save_checkpoint = True
        save_checkpoint = True
      else:
        save_checkpoint = True

      if save_checkpoint:
        if not gezi.get('tpu'):
          try:
            self.manager.save()
            self.logging(f'saved {self.ckpt_dir}/{self.checkpoint_name} at step {self.step}, with layers {num_layers}')
            saved = True
          except Exception as e:
            logging.warning(e)
            logging.warning('manager save checkpoint fail')
        else:
          # options for tpu, 没有manager控制不会删除老的checkpoint
          try:
            if tf.__version__ >= '2.3':
              old_checkpoint = tf.train.latest_checkpoint(self.ckpt_dir)
              options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
              self.logging(f'saved {self.ckpt_dir}/{self.checkpoint_name} with experimental_io_device=/job:localhost at step {self.step} with layers {num_layers}')
              self.ckpt.save(f'{self.ckpt_dir}/{self.checkpoint_name}', options=options)
              if old_checkpoint:
                self.logging(f'delete old checkpoint:{old_checkpoint}')
                # gezi.try_remove(old_checkpoint)
                os.system(f'rm -rf {old_checkpoint}*')
              saved = True
          except Exception:
            # logging.warning(traceback.format_exc())
            logging.warning('chekpoint save fail:', f'[{self.ckpt_dir}/{self.checkpoint_name}]')
  
    saved_model_h5 = False
    saved_model_done = False
    # 一把情况第一次save 也save .h5 为了检测save是否正常 后续等到最后一次才再次save h5 (tpu环境下 本地环境还是都是save tpu环境为了避免过多写drive)
    if (FLAGS.save_keras_model and (FLAGS.save_inter_models or is_last or (not gezi.get('tpu')))) or (not saved) or (self.saved_step is None):
      # model.save_weights h5 or not (check point)
      # model.save h5 or save_formt=tf (saved model) with graph
      # if FLAGS.save_graph and is_last:

      if FLAGS.save_graph:
        # HACK 这里暂时由于Warmp decay save之后载入会报错。。。 或者比较安全 model.copy() ? 不过这里限定必须is_last 也就是最后一个step训练完成之后
        # TODO 现在问题就是如果训练过程中产出的模型 拿过去不能直接用 因为只save weights了 需要依赖源代码
        model = copy.copy(self.model)
        if not FLAGS.save_graph_slim:
          try:
            optimizer = gezi.get('info')['optimizer']
            if FLAGS.fuxian_old_lr:
              optimizer._set_hyper('learning_rate', 0.001)
            model.compile(
                optimizer=optimizer,
                loss=gezi.get('info')['loss_fn'],
                metrics=gezi.get('info')['metrics']
            )
          except Exception:
            model.compile(
                optimizer='sgd',
                loss=gezi.get('info')['loss_fn'],
                metrics=gezi.get('info')['metrics']
            )
            pass
        else:
          # 模型大小并没有改变
          try:
            model.compile(
                optimizer='sgd',
                loss=gezi.get('info')['loss_fn']
                )
          except Exception:
            pass

        try:
          # to functional if possible
          if hasattr(model, 'get_model'):
            try:
              model = model.get_model()
            except Exception as e:
              logging.debug(e)
              logging.debug('model.get_model(to functinal) fail')
          try:
            model = tfmot.sparsity.keras.strip_pruning(model)
          except Exception:
            pass
          try:
            model.save(f'{FLAGS.model_dir}/model.h5')
          except Exception:
            model.save(f'{FLAGS.model_dir}/model.h5', save_format='tf')
          # tf.keras.models.save_model(model, f'{FLAGS.model_dir}/model.h5', include_optimizer=False)
          self.logging(f'saved {FLAGS.model_dir}/model.h5 with graph at step {self.step} with layers {num_layers}')
          saved_model_h5 = True
          saved = True
        except Exception as e:
          logging.debug(e)
          logging.debug(f'saved {FLAGS.model_dir}/model.h5 with graph at step {self.step} fail')
          if FLAGS.debug:
            logging.warning(traceback.format_exc())
          if FLAGS.saved_model:
            try:
              model.save(f'{FLAGS.model_dir}/saved_model')
              self.logging(f'saved {FLAGS.model_dir}/saved_model with graph at step {self.step}')
              saved_model_done = True
              saved = True
            except Exception as e:
              logging.warning(e)
              logging.warning(f'saved {FLAGS.model_dir}/model.h5 with graph at step {self.step} fail')
              pass
      
      # 如果model.save h5成功 同样可以 load_weights(model.h5)
      if not saved_model_h5:
        try:
          self.model.save_weights(f'{FLAGS.model_dir}/model.h5')
          self.logging(f'saved {FLAGS.model_dir}/model.h5 weights only at step {self.step} with layers {num_layers}')
          saved = True
        except Exception as e:
          logging.warning(e)
          logging.warning(f'saved {FLAGS.model_dir}/model.h5 weights only at step {self.step} fail')
          if FLAGS.debug:
            logging.warning(traceback.format_exc())
          # File system scheme '[local]' not implemented (file: '../working/v6/bonlime.DeeplabV3Plus.scale.drop03.100epoch/model.bin_temp_3ce99291b45c4bae9b06bc42b188169a/part-00000-of-00001')
	        # Encountered when executing an operation using EagerExecutor. This error cancels all future operations and poisons their output tensors.
          ## 2020-1205 不再尝试生成model.bin 因为似乎结果类似checkpoint而且载入可能有问题
          # if not gezi.get('tpu'):
          #   try:
          #     self.model.save_weights(f'{FLAGS.model_dir}/model.bin')
          #     self.logging(f'saved {FLAGS.model_dir}/model.bin weights only at step {self.step}')
          #     saved = True
          #   except Exception as e:
          #     logging.warning(e)
      elif FLAGS.save_weights:
        try:
          self.model.save_weights(f'{FLAGS.model_dir}/weights.h5')
          self.logging(f'saved {FLAGS.model_dir}/weights.h5 weights only at step {self.step} with layers {num_layers}')
          saved = True
        except Exception as e:
          logging.warning(e)
          logging.warning(f'saved {FLAGS.model_dir}/weights.h5 weights only at step {self.step} fail')
          if not gezi.get('tpu'):
            try:
              self.model.save_weights(f'{FLAGS.model_dir}/weights.bin')
              self.logging(f'saved {FLAGS.model_dir}/weights.bin weights only at step {self.step}')
              saved = True
            except Exception as e:
              logging.warning(e)
              logging.warning(f'saved {FLAGS.model_dir}/weights.bin weights only at step {self.step} fail')

    # 如果save_graph上面model.h5是吧 model.save生成的和下面是一样的saved model形式
    if FLAGS.saved_model and not saved_model_done:
      try:
        ## saved_model比较占空间 奇怪的是data部分也比较大 看上去像是有opimizer尽管include_optimizer设置False了 但是还是17M->48M类似这样
        ## 另外网络结构也占比较大 pb 可能16M 整体大小 64M 而model.h5尽管save没有设置 include_optimizer=False 但是实际大小只有17M 类别checkpoint会有48M
        ## saved model格式的一个好处是custom layer 不需要依赖原始代码 可以直接 tf.keras.models.load_model载入整个网络图+权重 而.h5不行
        ## 如果都是tf的标准layer没关系 但是如果有自定义custom layer载入h5 .load_model就需要传递custom_objects 
        ## https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects
        ## varaible比较大的问题有一个HACK WORKAROUND  
        ## sh ./train/v6/effdet.sh --mn=test3 --saved_model --steps=-1 类似这样重新载入再save之后大小就正常了
        # ValueError: _wrapped_model() should not modify its Python input arguments. Check if it modifies any lists or dicts passed as arguments. Modifying a copy is allowed.
        # tf.saved_model.save(self.model, f'{FLAGS.model_dir}/saved_model')
        self.model.save(f'{FLAGS.model_dir}/saved_model', include_optimizer=False)
        self.logging(f'saved {FLAGS.model_dir}/saved_model at step {self.step}')
        saved = True
      except Exception as e:
        logging.warning(e)

    if saved:
      self.saved_step = self.step
      gezi.write_to_txt(self.saved_step, f'{FLAGS.model_dir}/model_step.txt')
      # total step 各个loop工具最后会自己加 不要重复添加了 NOTICE
      # gezi.write_to_txt(self.saved_step, f'{FLAGS.model_dir}/total_step.txt')
      # here gcs_dest = {gcs_root}/{wandb_id}
      if FLAGS.gcs_dest:
        os.system(f'gsutil rm -r {FLAGS.gcs_dest}')
        gezi.gcs_cp(['*model*', 'checkpoint'], FLAGS.gcs_root, FLAGS.model_dir, FLAGS.wandb_id, verbose=0)

  def _is_save_step(self, step):
    steps_per_save = max(math.ceil(FLAGS.save_interval_epochs * self.steps_per_epoch), 1)
    save_ok = step % steps_per_save == 0 or step == self.steps_per_epoch * FLAGS.num_epochs
    
    if FLAGS.first_interval_epoch > 0:
      steps_save_first = max(math.ceil(FLAGS.first_interval_epoch * self.steps_per_epoch), 1)
      save_ok_first = step == steps_save_first
    else:
      save_ok_first = False
    
    if FLAGS.second_interval_epoch > 0:
      steps_save_second = max(math.ceil(FLAGS.second_interval_epoch * self.steps_per_epoch), 1)
      save_ok_second = step == steps_save_second
    else:
      save_ok_second = False  
    
    save_ok = save_ok or save_ok_first or save_ok_second
    
    return save_ok

  def is_save_step(self, step):
    pre_step = step - FLAGS.steps_per_execution
    for step_ in reversed(range(pre_step + 1, step + 1)):
      if self._is_save_step(step_):
        return step_
    return None

  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    if FLAGS.save_interval_epochs and FLAGS.save_interval_epochs > 0:
      step = self.is_save_step(self.step)
      if step is not None:
        logging.debug('save on batch end step:', self.step)
        is_last = self.step >= self.steps_per_epoch * FLAGS.num_epochs
        self.save(is_last)

  def on_train_end(self, logs={}):
    # 如果在on_batch_end没有最后一个step save 或者没有save成功比如tpu环境失败 这里在train的最后再次尝试save
    if FLAGS.save_model and self.step != self.saved_step:
      logging.debug('save on train end step:', self.step)
      self.save(is_last=True)

class TimerCallback(Callback):
  def __init__(self):
    self.timer = gezi.Timer()
    self.timer2 = gezi.Timer(reset=False)
    self.step = melt.get_total_step()
    self.logger = melt.get_summary_writer(set_walltime=False)
    self.last_step = 0
    self.writer = gezi.DfWriter(FLAGS.log_dir, filename='history.csv')
        
  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    if FLAGS.interval_steps and self.step % FLAGS.interval_steps == 0 or (self.step in [1, 100, 200]):
      steps = self.step - self.last_step
      self.last_step = self.step
      elapsed = self.timer.elapsed()
      hours_per_epoch = self.steps_per_epoch / steps * elapsed / 3600 if steps * elapsed else 0
      mintues_per_epoch = hours_per_epoch * 60
      epoch_time_info = '1epoch:[{:.1f}h]'.format(hours_per_epoch) if hours_per_epoch > 1 else  '1epoch:[{:.1f}m]'.format(mintues_per_epoch)
      steps_per_second =  steps / elapsed if elapsed else 0
      epoch = self.step / self.steps_per_epoch if self.steps_per_epoch else 0

      # TODO
      args = []
      if FLAGS.loop_train:
        args += ['train_hour:%s' % (FLAGS.train_hour)]
      devices = 'gpus' if not gezi.get('tpu') else 'tpus'
      args += [
              'epoch:%.2f/%d' % (epoch, FLAGS.epochs), 
              'step:%5d' % self.step, 
              'elap:[%.2f]' % elapsed,
              'batch:[%d]' % melt.batch_size(),
              '%s:[%d]' % (devices, FLAGS.num_gpus or 0), 
              'steps/s:[%.1f]' % steps_per_second,
              'insts/s:[%s]' % np.format_float_scientific(steps_per_second * melt.batch_size(), precision=1, trim='0'),
              '%s' % epoch_time_info,
            ]

      if 'loss' in logs:
        args += ['train:[%.4f]' % logs['loss']]

      if 'val_loss' in logs:
        args += ['valid:[%.4f]' % logs['val_loss']]

      lr = FLAGS.learning_rate
      
      optimizer = self.model.optimizer if not gezi.get('optimizers') else gezi.get('optimizers')[0]
      if hasattr(optimizer.lr, 'numpy'):
        lr = optimizer.lr.numpy()
      elif isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        # WarmUp
        lr = optimizer.lr(optimizer.iterations).numpy()
      else:
        lr = optimizer.lr

      # ic('-----------------', self.step, lr)

      args += ['lr:[%s]' % np.format_float_scientific(lr, precision=1, trim='0')]

      # to log.txt not log.html
      logging.info2(*args)

      logger = self.logger
      step = self.step

      if logger:
        logger.scalar('Others/batch_size_per_gpu', FLAGS.batch_size, step)
        logger.scalar('Others/batch_size', melt.batch_size(), step)
        logger.scalar('Others/num_gpus', FLAGS.num_gpus, step)
        logger.scalar('Others/max_lr', FLAGS.learning_rate, step)
        logger.scalar('Others/epochs', FLAGS.epochs, step)
        logger.scalar('History/learning_rate', lr, step)
        logger.scalar('Perf/steps_per_second', steps_per_second, step)
        logger.scalar('Perf/instances_per_second', steps_per_second * melt.batch_size(), step)
        logger.scalar('Perf/hours_per_epoch', hours_per_epoch, step)
        logger.scalar('Others/epoch', epoch, step)
        for key in logs:
          key_ = key
          if key == 'loss':
            key_ = 'train_loss'
          logger.scalar(f'History/{key_}', logs[key], step)
    
      res = {
        'round': FLAGS.round,
        'step': step,
        'epoch': epoch,
        'lr': lr,
        # 'train_loss': logs.get('loss', np.nan),
        # 'val_loss': logs.get('val_loss', np.nan),
      }
      for key in logs:
        key_ = key
        if key == 'loss':
          key_ = 'train_loss'
        res[key_] = logs[key]

      perf = {
        'insts_per_second': steps_per_second * melt.batch_size(),
        'steps_per_scond': steps_per_second,
        'hours_per_epoch': hours_per_epoch,
        'elapsed': self.timer2.elapsed(),
        'step': step,
      }

      for key in logs:
        if key.startswith('val_'):
          res[key] = logs[key]
      
      self.writer.append(res)
      if HAS_WANDB and FLAGS.wandb:
        res = gezi.dict_prefix(res, 'History/')
        perf = gezi.dict_prefix(perf, 'Perf/')
        wandb.log(perf, commit=False)
        wandb.log(res)

class UtilsCallback(Callback):
  def __init__(self):
    self.timer = gezi.Timer()
    self.step = melt.get_total_step()
    self.logger = None

  def work(self):
    model = self.model
    # if hasattr(model, 'init_predict'):
    #   model.init_predict()
    # #model.summary(print_fn=logging.info)
    if FLAGS.round == 0:
      melt.print_model(model, print_fn=logging.info)
    # total_params = sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables)
    total_params = model.count_params()
    l2 = melt.get_l2_sum(model) / total_params
    logging.info('Model total training parameters is:', total_params, 'with initial l2:', l2)
    self.l2 = l2

  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    if tf.executing_eagerly() and self.step == 1: 
      self.work()

  def on_train_begin(self, logs={}):
    if not tf.executing_eagerly():
      self.work()
      if self.logger:
        self.logger.scalar('loss/l2', self.l2, 0)

# FIXME not work on melt.Model wrap a functional model which inputs is dict
# reproduce ai/naic2020_seg/src/train.py
# ValueError: You tried to call `count_params` on model, but the layer isn't built. You can build it manually via: `model.build(batch_input_shape)`.
class L2LossCallback(Callback):
  def __init__(self, submodel=None):
    self.step =  gezi.get('eval_step') or melt.get_eval_step(from_file=True)
    # self.total_step = melt.get_total_step()
    self.logger = melt.get_summary_writer() if FLAGS.write_summary else None
    self.submodel = submodel

  def calc_l2(self, step=None, submodel=None):
    submodel = submodel or self.submodel
    model = submodel or self.model 
    try:
      total_params = model.count_params()
      l2 = melt.get_l2_sum(model) / total_params
    except Exception:
      total_params = 0
      l2 = 0.
    # print(' l2:{:e}'.format(l2))

    step = self.step + 1 if step is None else step
    logging.debug(f'Model {model} total training parameters is:', total_params, 'with l2:', l2, ' at step', step)
    if self.logger:
      self.logger.scalar('Others/total_params', total_params, step)
      self.logger.scalar('Others/model_size', total_params * 4 / (1024 * 1024), step)
      self.logger.scalar('Others/total_params', step, step)
      is_tpu = 1 if gezi.get('tpu') else 0
      self.logger.scalar('Others/is_tpu', is_tpu, step)
      tag = 'Others/l2' if not submodel else f'Others/l2/{submodel.name}'
      self.logger.scalar(tag, l2, step)
      
    if HAS_WANDB and FLAGS.wandb:
      res = {
              'Others/l2_loss': l2,
              'Others/total_params': total_params,
              'Others/model_size': total_params * 4 / (1024 * 1024), # model_size by M
              'Others/step': step,
            }
      wandb.log(res, commit=False)

  def on_train_begin(self, logs={}):
    if FLAGS.round == 0 and self.step == 0:
      self.calc_l2(0)

  def on_train_end(self, logs={}):
    self.calc_l2()

def _prepare_test_part(part, parts):
  try:
    files = gezi.get('info')['test_inputs']
  except Exception:
    files = gezi.list_files(FLAGS.test_input)
  # 假定20 file， 6 part 最后part 是 files[20:20] 空 等于全量数据了。。这时应该check 报错 强制使用设置5个worker
  start, end = gezi.get_fold(len(files), parts, part)
  Dataset = gezi.get('info')['Dataset']
  ds = Dataset('test')
  files = files[start:end]
  assert files, 'use one less worker'
  dataset = ds.make_batch(FLAGS.eval_batch_size, files)
  num_examples = len(ds)
  steps = -(-num_examples // FLAGS.eval_batch_size)
  return dataset, steps, num_examples

class TestCallback(Callback):
  def __init__(self, model, dataset, info_dataset=None, steps=None, num_examples=None, ofile=None, onames=[], 
               write_fn=None, inference_fn=None):
    self.step = melt.get_total_step()
    self.test_step = gezi.get('eval_step', 0) 
    self.strategy = melt.distributed.get_strategy()
    self.dataset = dataset
    # self.dataset = self.strategy.experimental_distribute_dataset(dataset)
    self.info_dataset = info_dataset 
    self.steps = steps
    self.num_test_examples = num_examples
    self.ofile = ofile if ofile else 'submission.csv'
    self.onames = onames if onames else FLAGS.test_names
    if not self.onames:
      self.onames = ['id', 'pred']
    self.write_fn, self.write_streaming_fn = None, None
    if not FLAGS.infer_write_streaming:
      self.write_fn = write_fn 
    else:
      self.write_streaming_fn = write_fn
    self.model = model
    self.eval_keys = model.eval_keys if hasattr(model, 'eval_keys') else []
    self.eval_keys = gezi.get('test_keys') or self.eval_keys
    self.out_keys = model.out_keys if hasattr(model, 'out_keys') else []
    self.cached_xs = None
    self.inference_fn = inference_fn

    if self.info_dataset is not None and isinstance(model, melt.Model):
      find_oof = False
      example = next(iter(self.dataset))[0]
      for key in self.eval_keys:
        if key not in example:
          logging.debug(f'Missing {key} in example so need info_dataset in TestCallback')
          find_oof = True
          break
      if not find_oof:
        self.info_dataset = None
    logging.debug('self.info_dataset in TestCallback: ', self.info_dataset)

  def predict(self):
    learning_phase = K.learning_phase()
    K.set_learning_phase(0)
    model = self.model
    ys = []
    preds = []
    infos = None
   
    # TODO 有一些冗余 test不像eval 似乎只需要output和id就可以了 不需要eval 计算依赖的其他x[key]
    if self.info_dataset is None:
      assert isinstance(model, melt.Model) 
      outputs = model.infer(self.dataset, steps=self.steps, dump_inputs=True, desc='test_predict_all', verbose=0, leave=FLAGS.test_leave, write_fn=self.write_streaming_fn)
      if not isinstance(outputs, dict):
        outputs = {'pred': outputs}
      xs = outputs
    else:
      if FLAGS.predict_on_batch:
        tmp = {}
        outputs = {}
        test_iter = iter(self.info_dataset)
        for i in tqdm(range(self.steps), ascii=True, desc='test_predict_on_batch', leave=FLAGS.test_leave):
          xs, _ = next(test_iter)
          # xs = tonumpy(xs)
          res = model.predict_on_batch(xs)
          if not isinstance(res, dict): 
            res = {'pred': res}
          for key in xs:
            if self.eval_keys and key not in self.eval_keys:
              continue
            xs[key] = xs[key].numpy()
            if key not in tmp:
              tmp[key] = [xs[key]]
            else:
              tmp[key].append(xs[key])
          for key in res:
            if key not in outputs:
              outputs[key] = [res[key]]
            else:
              outputs[key].append(res[key])

        for key in tmp:
          tmp[key] = np.concatenate(tmp[key]) 
        xs = tmp
        for key in outputs:
          outputs[key] = np.concatenate(outputs[key])
      else:
        if isinstance(model, melt.Model) and not FLAGS.keras_loop:
          outputs = model.infer(self.dataset, steps=self.steps, dump_inputs=False, desc='test_predict', verbose=0, leave=FLAGS.test_leave, write_fn=self.write_streaming_fn)
        else:
          outputs = model.predict(self.dataset, steps=self.steps, callbacks=[TQDMProgressBar('test_predict')], verbose=0)
        if not isinstance(outputs, dict):
          outputs = {'pred': outputs}

        if self.cached_xs is None:
          if isinstance(model, melt.Model):
            xs = model.loop(self.info_dataset, steps=self.steps, desc='test_loop', verbose=0, leave=FLAGS.test_leave)
          else:
            # functional keras model
            # logging.info('test_loop')
            # info_dataset = self.info_dataset.map(lambda x, y: x).unbatch()
            # xs = next(iter(info_dataset.batch(self.num_test_examples)))
            # for key in xs:
            #   xs[key] = xs[key].numpy()
            tmp = {}
            test_iter = iter(self.info_dataset)
            for i in tqdm(range(self.steps), ascii=True, desc='test_loop', leave=FLAGS.test_leave):
              xs, _ = next(test_iter)
              if not isinstance(xs, dict):
                xs = {'pred': xs}
              for key in xs:
                if self.eval_keys and not key in self.eval_keys:
                  continue
                xs[key] = xs[key].numpy()
                if key not in tmp:
                  tmp[key] = [xs[key]]
                else:
                  tmp[key].append(xs[key])

            for key in tmp:
              tmp[key] = np.concatenate(tmp[key]) 
            xs = tmp

          if FLAGS.cache_test_input:
            self.cached_xs = xs
        else:
          xs = self.cached_xs

    for key in outputs:
      outputs[key] = outputs[key][:self.num_test_examples]
    
    for key in xs:
      xs[key] = xs[key][:self.num_test_examples]

    assert 'pred' in outputs, 'there must have key:pred in outputs'
    preds = outputs['pred']
    preds = gezi.squeeze(preds)
    self.preds = preds

    self.x = dict(zip(self.eval_keys, [gezi.squeeze(xs[key]) for key in self.eval_keys if key in xs]))
    self.other  = dict(zip(self.out_keys, [gezi.squeeze(outputs[key]) for key in self.out_keys if key in outputs]))

    # for key in self.x:
    #   self.x[key] = gezi.decode(self.x[key])
      
    # for key in self.other:
    #   self.other[key] = gezi.decode(self.other[key])
    
    K.set_learning_phase(learning_phase)
    return preds

  def test(self):
    if self.dataset is None:
      logging.warning('test dataset is None, check your test_input FLAGS.test_input:', FLAGS.test_input)
      return

    if FLAGS.ema_inject:
      ema = gezi.get('ema') # set in husky.train
      ema.apply_ema_weights() # 将EMA的权重应用到模型中

    if FLAGS.opt_ema or FLAGS.opt_swa:
      non_avg_weights = self.model.get_weights()
      optimizer = gezi.get('optimizer')
      optimizer.assign_average_vars(self.model.variables)
      # result is currently None, since `super._save_model` doesn't
      # return anything, but this may change in the future.
      # result = super()._save_model(epoch, logs)
      # self.model.set_weights(non_avg_weights)

    # wandb 可能process die 如果test的tqdm较长时间loop TODO 再验证是否需要这里finish以及 能否wandb多次finish
    if not FLAGS.wandb_test:
      run = gezi.get('wandb_run')
      if run:
        try:
          run.finish()
        except Exception as e:
          logging.warning(e)
        gezi.set('wandb_run', None)

    model = self.model
    model.mode = 'test'
    # timer = gezi.Timer()

    ofile = f'{FLAGS.model_dir}/{self.ofile}'
    if self.inference_fn:
      logging.info('predict and write test result to', ofile)
      args = inspect.getargspec(self.inference_fn).args 
      kwargs = {}
      if 'model' in args:
        kwargs['model'] = model
      if 'steps' in args:
        kwargs['steps'] = self.steps
      if 'ofile' in args:
        kwargs['ofile'] = ofile
      if 'outdir' in args:
        kwargs['outdir'] = os.path.dirname(ofile)
      if 'num_examples' in args:
        kwargs['num_examples'] = self.num_test_examples
      if 'desc' in args:
        kwargs['desc'] = 'inference'
      dataset = self.dataset

      if FLAGS.parts and not FLAGS.use_shard:
        dataset, steps, num_test_examples = _prepare_test_part(FLAGS.part, FLAGS.parts)
        if 'steps' in kwargs:
          kwargs['steps'] = steps
        if 'num_examples' in kwargs:
          kwargs['num_examples'] = num_test_examples
      
      if FLAGS.parts:
        if 'desc' in kwargs:
          kwargs['desc'] = f'inference: {FLAGS.part}/{FLAGS.parts}'

      # strategy = melt.distributed.get_strategy()
      # dataset = strategy.experimental_distribute_dataset(self.dataset)
      self.inference_fn(dataset, **kwargs)
      # strategy.run(self.inference_fn, (dataset, model, steps, ofile))
      if FLAGS.ema_inject:
        ema.reset_old_weights() 
      if FLAGS.opt_ema or FLAGS.opt_swa:
        self.model.set_weights(non_avg_weights)
      return

    with gezi.Timer('test_pred_loop', False, print_fn=logging.debug):
      preds = self.predict()

    if FLAGS.ema_inject:
      ema.reset_old_weights() 
    if FLAGS.opt_ema or FLAGS.opt_swa:
      self.model.set_weights(non_avg_weights)

    if FLAGS.work_mode != 'train':
      from numba import cuda
      gpus = gezi.get_global('gpus')
      if gpus:
        for gpu in gpus:
          try:
            cuda.select_device(gpu)
            cuda.close()
          except Exception:
            logging.warning(traceback.format_exc())
            logging.warning(f'cuda select_device gpu:{gpu} fail, gpus:{gpus}, CUDA_VISIBLE_DIVICES:{gezi.get_specific_gpus()}')
  
    # elapsed = timer.elapsed()
    # run = gezi.get('wandb_run')
    # if run:
    #   run.log({''})

    logging.info('write test result to', ofile)
    if not self.write_fn:
      x = {}
      if 'id' in x:
        x[self.onames[0]] = self.x['id']
      else:
        x.update(self.x)
      x[self.onames[1]] = preds

      df = pd.DataFrame(x)
      try:
        df.id = df.id.astype(int)
      except Exception:
        pass

      if 'id'  in x:
        df = df.sort_values(['id'])

      self.test_step += 1
      step = self.test_step if FLAGS.fold is None else FLAGS.fold
      df.to_csv(ofile, index=False)
      # if FLAGS.fold is None:
      #   df.to_csv(f'{FLAGS.model_dir}/infos/{self.ofile}.csv', index=False) 
    else:
      kwargs_write = {}
      write_args = inspect.getfullargspec(self.write_fn).args 
      if 'others' in write_args:
        kwargs_write['others'] = self.other
      self.write_fn(self.x, preds, ofile, **kwargs_write)

  def on_train_end(self, logs={}):
    # if FLAGS.do_test and FLAGS.test_input:
    if FLAGS.do_test:
      # TODO 现在看起来 只在训练最后做一次test 就ok吧 用do_test=False或者FLAGS.test_interval_epochs=-1来屏蔽
      # 目前keras的test和valid只支持简单输出 TODO 使用自定义输出函数
      if FLAGS.test_interval_epochs and FLAGS.test_interval_epochs > 0:
        self.test()

  def on_train_begin(self, logs={}):
    if FLAGS.mode == 'test':
      self.test()
      exit(0)
   
  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
