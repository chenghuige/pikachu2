#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2019-09-04 17:29:10.942618
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

from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import math
import inspect
import traceback
import copy
# from tqdm import tqdm
# from tqdm.notebook import tqdm
from gezi import tqdm
import pandas as pd
from collections import OrderedDict

from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import multiprocessing

import numpy as np 
import gezi 
logging = gezi.logging
import melt
from melt.flow.flow import _try_eval, _on_epoch_end, _async_valid, _try_eval_day
from melt.distributed import tonumpy
from husky.callbacks.tqdm_progress_bar import TQDMProgressBar

try:
  import wandb
  HAS_WANDB = True
except ImportError:
  HAS_WANDB = False
  pass

def _prepare_eval_part(part, parts):
  try:
    files = gezi.get('info')['valid_inputs']
  except Exception:
    files = gezi.list_files(FLAGS.valid_input)
  start, end = gezi.get_fold(len(files), parts, part)
  Dataset = gezi.get('info')['Dataset']
  ds = Dataset('valid')
  files = files[start:end]
  assert files, 'use one less worker'
  dataset = ds.make_batch(FLAGS.eval_batch_size, files)
  num_examples = len(ds)
  steps = -(-num_examples // FLAGS.eval_batch_size)
  return dataset, steps, num_examples

# TODO wandb 不支持异步新启动的进程 可以考虑 ValidLossCallback 进行valid计算的 batch end 检查文件 写入wandb metrics信息, 当然一般不需要异步
def eval(eval_fn, y, y_, dataset, kwargs, logger, writer, eval_step, step, epoch, 
        num_valid_examples, is_last, eval_pred_time, 
        gtimer=None, pre=None, logs={}, wandb_run=None, silent=False):
  timer = gezi.Timer('eval')
  melt.save_eval_step()
  # 异步需要重新生成一个logger避免同时写
  if logger:
    logger = melt.get_summary_writer()
  writer = gezi.DfWriter(FLAGS.log_dir, filename='metrics.csv')
  writer2 = gezi.DfWriter(FLAGS.log_dir, filename='infos.csv')
  writer3 = gezi.DfWriter(FLAGS.log_dir, filename='others.csv')
  # eval_step += 1
  # gezi.set('eval_step', eval_step)

  # TODO 当前这种设计存在一个问题就是如果eval_fn比较慢就迟迟不能展现eval结果 比如eval内部需要写图片log比较慢这种
  # 需要阻塞等待处理完在处理res结果 而实际上res早就完成一直等待 不过这么处理逻辑比较简单清晰
  try:
    if y is not None:
      results = eval_fn(y, y_, **kwargs)
    else:
      # strategy = melt.distributed.get_strategy()
      # dataset = strategy.experimental_distribute_dataset(dataset)
      if FLAGS.parts and not FLAGS.use_shard:
        dataset, steps, num_valid_examples = _prepare_eval_part(FLAGS.part, FLAGS.parts)
        if 'steps' in kwargs:
          kwargs['steps'] = steps
        if 'num_examples' in kwargs:
          kwargs['num_examples'] = num_valid_examples
      if FLAGS.parts:
        if 'desc' in kwargs:
          kwargs['desc'] = f'eval: {FLAGS.part}/{FLAGS.parts}'

      results = eval_fn(dataset, **kwargs)
  except Exception:
    logging.warning('eval fn error')
    logging.warning(traceback.format_exc())
    results = {}
  logging.debug(f'eval_step: {eval_step} step: {step}', 'epoch: %.2f' % epoch)

  if isinstance(results, (list, tuple)):
    try:
      names, vals = results 
    except Exception:
      raise ValueError(results)
    results = OrderedDict(zip(names, vals))

  if results:
    if not list(results.keys())[0].startswith('Metrics/'):
      results = gezi.dict_prefix(results, 'Metrics/')
  
  results['insts'] = len(y) if y is not None else num_valid_examples
  results['insts_per_second'] = results['insts'] / eval_pred_time
  
  results['elapsed'] = gtimer.elapsed(reset=False)
  results['eval_metrics_time'] = timer.elapsed()
  results['eval_pred_time'] = eval_pred_time
  
  res = type(results)([(key, results[key]) for key in results if key.startswith('Metrics/')])
  res2 = type(results)([(key, results[key]) for key in results if key.startswith('Infos/')])
  res3 = type(results)([(key, results[key]) for key in results if not (key in res or key in res2) ])

  res = gezi.dict_rename(res, 'Metrics/', '')
  res2 = gezi.dict_rename(res2, 'Infos/', '')
  res3 = gezi.dict_rename(res3, 'Others/', '')

  # print Metrics Info and all others to logfile
  logging.debug('Metrics:', gezi.FormatDict(res))
  if res2:
    logging.debug('Infos:', gezi.FormatDict(res2))
  if res3:
    logging.debug('Others:', gezi.FormatDict(res3))
  
  # only print Metrics and if less then 15
  print_fn = logging.info if (is_last or FLAGS.eval_verbose > 0) else logging.debug
  # print_fn = logging.info
  max_show = 13

  desc = f'[{FLAGS.model_name}] eval_step: {eval_step}'
  logs = gezi.get('logs')
  if logs:
    if not FLAGS.evshow_loss_only:
      logs = dict([(key, f'{val:.4f}') for key, val in logs.items()])
    else:
      logs = dict([(key, f'{val:.4f}') for key, val in logs.items() if 'loss' in key])
    desc += f' {logs}'
  
  if res and len(res) <= max_show:
    gezi.pprint(res, print_fn=print_fn, desc=desc, format='%.4f')
  elif res2 and len(res2) <= max_show:
    gezi.pprint(res, print_fn=print_fn, desc=desc, format='%.4f')
  else:
    res_ = type(res)(list(res.items())[:max_show])
    gezi.pprint(res_, print_fn=print_fn, desc=desc, format='%.4f')

  ## 不再写logs evaluate callback的信息通过 gezi.get('history') 获取
  # logs.update(res)

  # print(f'\rvalid_metrics: {list(zip(names, vals_))}  num examples: {len(y)} eval_step: {self.eval_step} step: {self.step}', 'epoch: %.1f' % self.epoch, end=100*' '+'\n')

  # for name, val in zip(names, vals):
  #   if pre:
  #     if logger:
  #       logger.scalar(f'{pre}/{name}', val, eval_step)
  #     gezi.get('history')[f'{pre}/name'].append(val)
  #   else:
  #     if name.startswith('Metrics/'):
  #       name = 'A' + name # to be first on tensorboard
  #     if logger:
  #       logger.scalar(name, val, eval_step)
  #     gezi.get('history')[name].append(val)

  if silent:
    return

  # if res3:
  res3['step'] = eval_step
  res3['epoch'] = epoch
  writer3.write(res3)
  gezi.set('Others', res3)
  if wandb_run:
    wandb_run.log(dict(gezi.dict_prefix(res3, 'Others/')), commit=False)
  if logger:
    logger.scalars(dict(gezi.dict_prefix(res3, 'Others/')), eval_step)

  if res2:
    res2['step'] = eval_step
    res2['epoch'] = epoch
    writer2.write(res2)
    gezi.set('Infos', res2)
    if wandb_run:
      wandb_run.log(dict(gezi.dict_prefix(res2, 'Infos/')), commit=False)
    if logger:
      logger.scalars(dict(gezi.dict_prefix(res2, 'Infos/')), eval_step)

  if not res:
    results['step'] = eval_step
    results['epoch'] = epoch
    writer.write(results)
  else:
    res['step'] = eval_step
    res['epoch'] = epoch
    res['insts'] = results['insts']
    writer.write(res)

  gezi.set('Metrics', res)
  if wandb_run:
    wandb_run.log(dict(gezi.dict_prefix(res, 'Metrics/')), commit=True)
  if logger:
    logger.scalars(dict(gezi.dict_prefix(res, 'Metrics/')), eval_step)

#   melt.save_eval_step()   
  _try_eval_day()


class EvalCallback(Callback):
  def __init__(self, model, dataset, eval_fn, 
               info_dataset=None,
               steps=None, 
               num_examples=None, 
               steps_per_epoch=None, 
               write_valid=None, 
               write_fn=None, 
               summary_writer=None, 
               loss_fn=None, 
               pre=None, 
               final_hook_fn=None,
               pretrained_dir=None):

    self.model = model
    self.strategy = melt.distributed.get_strategy()
    self.dataset = dataset
    # self.dataset = self.strategy.experimental_distribute_dataset(dataset) #如果是用标准model.fit evaluate predict 这里不需要转换 转换的话tf2.3ok 2.2还不行 可能2.3做了是否已经转换的判断了
    self.info_dataset = info_dataset 

    self.write_fn = write_fn
    self.loss_fn = loss_fn

    self.num_valid_examples = num_examples
    self.steps_per_epoch = steps_per_epoch
    self.write_valid_ = write_valid if write_valid is not None else FLAGS.write_valid

    self.y = None

    if not tf.executing_eagerly() and tf.__version__ < '2':
      self.sess = melt.get_session()
    else:
      self.sess = None

    self.steps = steps
    self.eval_keys = model.eval_keys if hasattr(model, 'eval_keys') else []

    self.out_keys = model.out_keys if hasattr(model, 'out_keys') else []
    
    self.preds = None
    self.x = None
    self.other= None

    self.eval_fn = eval_fn

    self.step = melt.get_total_step()

    if not os.path.exists(f'{FLAGS.model_dir}/eval_step.txt'):
      if pretrained_dir:
        eval_step_file = f'{pretrained_dir}/eval_step.txt'
        if os.path.exists(eval_step_file):
          gezi.copyfile(eval_step_file, f'{FLAGS.model_dir}/eval_step.txt')

    self.eval_step = gezi.get('eval_step') or melt.get_eval_step(from_file=True)
    self.epoch = 0.
    self.steps_per_eval = max(math.ceil(FLAGS.valid_interval_epochs * self.steps_per_epoch), 1)

    self.logger = None
    if FLAGS.log_dir and FLAGS.write_summary and FLAGS.write_metric_summary:
      self.logger = melt.get_summary_writer()

    self.writer =  gezi.get('metric_writer', gezi.DfWriter(FLAGS.log_dir))
    self.timer = gezi.Timer(reset=False)
    self.pre_step = self.step

    self.results = None
    self.write_valid_steps = set()

    self.cached_xs = None

    self.pre = pre

    self.final_hook_fn = final_hook_fn or gezi.get('final_hook_fn')

    if FLAGS.valid_hour:
      os.system(f'mkdir -p {FLAGS.model_dir}/infos/{FLAGS.valid_hour}')
      self.ofile = f'{FLAGS.model_dir}/infos/{FLAGS.valid_hour}/valid.csv'
    else:
      os.system(f'mkdir -p {FLAGS.model_dir}/infos')
      self.ofile = f'{FLAGS.model_dir}/infos/valid_{self.step}.csv'

    self.outdir = os.path.dirname(self.ofile)
    self.is_last = False

  def get_y_eager(self):
    model = self.model
    ys = []
    preds = []
    infos = None
    eval_step = self.eval_step + 1

    # dataset = iter(self.dataset)

    # l = []
    # for x, y in tqdm(gezi.get('info')['eval_dataset'], total=self.steps, desc='abc'):
    #   out = model(x)
    #   with tf.device('/cpu:0'):
    #     out = tf.identity(out)
    #   l.append(out)

    # print(len(l))
    # print(l[-1])
    # ic(self.info_dataset, FLAGS.predict_on_batch, isinstance(model, melt.Model), FLAGS.keras_loop)
    if self.info_dataset is None:
      assert isinstance(model, melt.Model) 
      outputs = model.infer(self.dataset, steps=self.steps, dump_inputs=True, desc=f'eval_predict_all_{eval_step}', verbose=0, leave=FLAGS.eval_leave)
      if isinstance(outputs, (tuple, list)):
        if len(outputs) == len(self.out_keys):
          outputs = dict(zip(self.out_keys, outputs))
        else:
            outputs = dict(zip(['pred'] + self.out_keys, outputs))
      elif not isinstance(outputs, dict):
        outputs = {'pred': outputs}
      xs = outputs
    else:
      if FLAGS.predict_on_batch:
        tmp = {}
        outputs = {}
        eval_iter = iter(self.info_dataset)
        for i in tqdm(range(self.steps), ascii=False, desc='eval_predict_on_batch', leave=FLAGS.eval_leave):
          xs, _ = next(eval_iter)
          # xs = tonumpy(xs)
          res = model.predict_on_batch(xs)
          if isinstance(res, (tuple, list)):
            if len(res) == len(self.out_keys):
              res = dict(zip(self.out_keys, res))
            else:
              res = dict(zip(['pred'] + self.out_keys, res))
          elif not isinstance(res, dict): 
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
          outputs = model.infer(self.dataset, steps=self.steps, dump_inputs=False, desc=f'eval_predict_{eval_step}', verbose=0, leave=FLAGS.eval_leave)
        else:
          outputs = model.predict(self.dataset, steps=self.steps, callbacks=[TQDMProgressBar(f'eval_predict_{eval_step}')], verbose=0)

        if isinstance(outputs, (tuple, list)):
          if len(outputs) == len(self.out_keys):
            outputs = dict(zip(self.out_keys, outputs))
          else:
            outputs = dict(zip(['pred'] + self.out_keys, outputs))
        elif not isinstance(outputs, dict):
          outputs = {'pred': outputs}

        if self.cached_xs is None:
          if isinstance(model, melt.Model):
            xs = model.loop(self.info_dataset, steps=self.steps, desc=f'eval_loop_{eval_step}', verbose=0, leave=FLAGS.eval_leave)
          else:
            # functional keras model
            # logging.info('test_loop')
            # info_dataset = self.info_dataset.map(lambda x, y: x).unbatch()
            # xs = next(iter(info_dataset.batch(self.num_test_examples)))
            # for key in xs:
            #   xs[key] = xs[key].numpy()
            tmp = {}
            info_iter = iter(self.info_dataset)
            for i in tqdm(range(self.steps), ascii=False, desc=f'eval_loop_{eval_step}', leave=FLAGS.eval_leave):
              xs, y = next(info_iter)
              if not isinstance(xs, dict):
                xs = {'pred': xs}
              xs['y'] = y
              for key in xs:
                if self.eval_keys and not key in self.eval_keys and key != 'y':
                  continue
                xs[key] = xs[key].numpy()
                if key not in tmp:
                  tmp[key] = [xs[key]]
                else:
                  tmp[key].append(xs[key])

            for key in tmp:
              tmp[key] = np.concatenate(tmp[key]) 
            xs = tmp

          if FLAGS.cache_valid_input:
            self.cached_xs = xs
        else:
          xs = self.cached_xs

    assert 'pred' in outputs, 'there must have key:pred in outputs'
    preds = outputs['pred']
    if not 'y' in xs:
      ys = preds
    else:
      ys = xs['y']

    self.y = gezi.squeeze(ys)[:self.num_valid_examples]
    self.preds = gezi.squeeze(preds)[:self.num_valid_examples]
    
    if not (hasattr(model, 'remove_pred') and model.remove_pred):
      assert len(self.y) == self.num_valid_examples, f'{len(self.y)} {self.num_valid_examples}'
      assert len(self.preds) == self.num_valid_examples

    try:
      self.x = dict(zip(self.eval_keys, [gezi.squeeze(xs[key]) for key in self.eval_keys]))
    except Exception:
      logging.debug('xs.keys', xs.keys(), 'eval_keys', self.eval_keys)
      eval_keys = []
      for key in self.eval_keys:
        if key in xs:
          eval_keys.append(key)
        else:
          logging.debug(f'{key} will be excluded from eval_keys')
      self.eval_keys = eval_keys
      self.x = dict(zip(self.eval_keys, [gezi.squeeze(xs[key]) for key in self.eval_keys]))
      
    self.other = dict(zip(self.out_keys, [gezi.squeeze(outputs[key]) for key in self.out_keys if key in outputs]))

    for key in self.x:
      self.x[key] = self.x[key][:self.num_valid_examples]
      
    for key in self.other:
      self.other[key] = self.other[key][:self.num_valid_examples]

    return self.y

  # just for test of speed TODO why her keras is much slower then tf only.. even below slower then tf predict ..
  def one_loop(self): 
    model = self.model
    self.sess.run(self.dataset.initializer)
    # with tf.device('/cpu:0'):
    for _ in tqdm(range(self.steps), total=self.steps, ascii=False, leave=FLAGS.eval_leave):
      x_t, y_t = self.dataset.get_next()
      #x = self.sess.run(model(x_t))
      y = self.sess.run(y_t)

  # depreciated for tf2 just use get_y_eager
  def get_y_graph(self):
    model = self.model
    self.sess.run(self.dataset.initializer)
    ys = []
    preds = []
    infos = [[] for _ in range(len(self.eval_keys))]
    other_tensors = self.out_hook(model) if self.out_hook else {}
    others_list = [[] for _ in range(len(other_tensors))]

    try:
      for _ in tqdm(range(self.steps), total=self.steps, ascii=False, leave=FLAGS.eval_leave):
        x_t, y_t = self.dataset.get_next()
        x_t_ ={} 
        for key in self.eval_keys:
          x_t_[key] = x_t[key]
      
        #x, y, others, y_ = self.sess.run([x_t_, y_t, other_tensors, model(x_t)])
        x, y, others = self.sess.run([x_t_, y_t, other_tensors])
        # if eval batch size 5120 here still 512... TODO FIXME
        y = y.squeeze()
        # y_ = y_.squeeze()
        ys.append(y)
        # preds.append(y_)
        for i, key in enumerate(self.eval_keys):
          infos[i].append(x[key].squeeze())
        for i, key in enumerate(others): 
          others_list[i].append(others[key].squeeze())
      preds = model.predict(self.dataset, steps=self.steps)
      preds = preds.squeeze()
    except tf.errors.OutOfRangeError:
      pass

    self.y = np.concatenate(ys, axis=0)[:self.num_valid_examples]
    # self.preds = np.concatenate(preds, axis=0)[:self.num_valid_examples]
    self.preds = preds[:self.num_valid_examples]
    self.x = dict(zip(self.eval_keys, [gezi.decode(np.concatenate(infos[i])[:self.num_valid_examples]) for i in range(len(self.eval_keys))]))
    others = [np.concatenate(others_list[i]) for i in range(len(others_list))][:self.num_valid_examples]
    other = {}
    for i, key in enumerate(other_tensors): 
      other[key] = others[i][:self.num_valid_examples]
    self.other = other
    return self.y

  def get_y(self):
    if tf.executing_eagerly() or tf.__version__ >= '2':
      return self.get_y_eager() 
    else:
      return self.get_y_graph()

  def eval(self, is_last=False, silent=False, logs={}):
    logging.debug('-----------eval is last', is_last)
    FLAGS.is_last_eval = is_last

    if self.dataset is None or self.eval_fn is None:
      ic(self.dataset, self.eval_fn)
      return

    if FLAGS.ema_inject:
      ema = gezi.get('ema') # set in husky.train
      ema.apply_ema_weights() # 将EMA的权重应用到模型中

    if FLAGS.opt_ema or FLAGS.opt_swa:
      non_avg_weights = self.model.get_weights()
      optimizer = gezi.get('optimizer')
      from tensorflow_addons.optimizers.average_wrapper import AveragedOptimizerWrapper
      if not isinstance(optimizer, AveragedOptimizerWrapper):
        raise TypeError(
            "AverageModelCheckpoint is only used when training"
            "with MovingAverage or StochasticAverage"
        )
      optimizer.assign_average_vars(self.model.variables)
      # result is currently None, since `super._save_model` doesn't
      # return anything, but this may change in the future.
      # result = super()._save_model(epoch, logs)
      # self.model.set_weights(non_avg_weights)

    args = inspect.getargspec(self.eval_fn).args 

    if is_last:
      # 可能设置 比如最后一个step 执行全部验证集合
      if FLAGS.full_validation_final:
        FLAGS.num_valid = 0
        self.steps = gezi.get('num_full_valid_steps_per_epoch') or self.steps
        self.num_valid_examples = gezi.get('num_full_valid_examples') or self.num_valid_examples
        self.cached_xs = None
      if self.final_hook_fn:
        self.final_hook_fn()

    pre = self.pre
    ## 及时最后一次验证也异步 保证不阻塞影响后面程序执行
#     if FLAGS.async_valid and FLAGS.work_mode != 'valid' and not is_last:
    if FLAGS.async_valid and FLAGS.work_mode != 'valid':
      _async_valid()
      return
      
    # if self.step > 0:
    #   elapsed = self.timer.elapsed()
    #   steps = self.step - self.pre_step
    #   self.pre_step = self.step
    #   steps_per_second = steps / elapsed
    #   insts_per_second = steps * melt.batch_size() / elapsed
    #   hours_per_epoch = (self.steps_per_epoch / steps_per_second) / 3600 if steps else 0
    #   # print('\n\r', 'steps/s: %.1f - insts/s: %.1f - 1epoch: %.2fh' % (steps_per_second, insts_per_second, hours_per_epoch)) 
    
    learning_phase = K.learning_phase()
    K.set_learning_phase(0)
    self.model.mode = 'valid'
    # strategy = melt.distributed.get_strategy()
    # with strategy.scope():
    with gezi.Timer('eval_pred_loop', False, print_fn=logging.debug) as pred_timer:
      y = self.get_y() if not 'dataset' in args else None
      pred_time = pred_timer.elapsed(reset=False)

    if y is None:
      FLAGS.async_eval = False
    else:
      pass
    
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
  
    # TODO check, we always write valid in parallel using multiprocess during train 
    if (self.write_valid_ or (FLAGS.write_valid_final and is_last)) and (not FLAGS.write_valid_after_eval):
#       if not FLAGS.async_eval:
#         self.write_valid()
#       else:
      # q = multiprocessing.Process(target=self.write_valid, args=('valid',))
      q = multiprocessing.Process(target=self.write_valid)
      q.start()

    p, q = None, None
    if not FLAGS.write_valid_only:
      y_ = self.preds
      eval_fn = self.eval_fn
    
      if not silent:
        # 假定async_valid就是每次都async 或者async_valid=False就是完全每次都同步
        # self.eval_step = melt.get_eval_step(from_file=True)
        self.eval_step += 1
        gezi.set('eval_step', self.eval_step)

      kwargs = {}   
      if 'info' in args:
        kwargs['info'] = self.x
      if 'x' in args:
        kwargs['x'] = self.x
      if 'model' in args:
        kwargs['model'] = self.model
      if 'other' in args:
        kwargs['other'] = self.other
      if 'others' in args:
        kwargs['others'] = self.other
      if 'eval_step' in args:
        kwargs['eval_step'] = self.eval_step
      if 'step' in args:
        kwargs['step'] = self.step
      if 'is_last' in args:
        kwargs['is_last'] = is_last
      if 'steps' in args:
        kwargs['steps'] = self.steps
      if 'loss_fn' in args:
        kwargs['loss_fn'] = self.loss_fn
      if 'ofile' in args:
        kwargs['ofile'] = self.ofile
      if 'outdir' in args:
        kwargs['outdir'] = self.outdir
      if 'num_examples' in args:
        kwargs['num_examples'] = self.num_valid_examples
      if 'return_dict' in args:
        kwargs['return_dict'] = True
      if 'desc' in args:
        kwargs['desc'] = 'eval'

      wandb_run = gezi.get('wandb_run')
      # is_last不需要再异步 另外就是v100 tione最后tf和multiprocess兼容有点问题
      if (not FLAGS.async_eval) or is_last:
        eval(eval_fn, y, y_, self.dataset, kwargs, self.logger, self.writer, self.eval_step, self.step, self.epoch, 
             self.num_valid_examples, self.is_last or is_last, pred_time, self.timer, pre, logs, wandb_run, silent)
      else:
        p = multiprocessing.Process(target=eval, args=(eval_fn, y, y_, self.dataset, kwargs, self.logger, self.writer, self.eval_step, self.step, self.epoch, 
                                                       self.num_valid_examples, self.is_last or is_last, pred_time, self.timer, pre, logs, wandb_run, silent))
        p.start()

      if (self.write_valid_ or (FLAGS.write_valid_final and is_last)) and FLAGS.write_valid_after_eval:
        self.write_valid()

      _try_eval_day() # 现在只是为了 fee/rank 做异步天级别验证 输入是24小时的所以valid.csv
      

    if is_last and (p or q) and not (FLAGS.test_input and FLAGS.do_test):
      logging.debug('Waiting async eval finish finally')
      if p:
        p.join()
      if q:
        q.join()

    if FLAGS.ema_inject:
      ema.reset_old_weights() 
    if FLAGS.opt_ema or FLAGS.opt_swa:
      self.model.set_weights(non_avg_weights)

    K.set_learning_phase(learning_phase)

  def write_valid(self, filename=None):
    if self.eval_step not in self.write_valid_steps:
      step = self.eval_step if FLAGS.fold is None else FLAGS.fold
      
      if FLAGS.valid_hour and '*' in FLAGS.valid_hour:
        if FLAGS.valid_input and not ',' in FLAGS.valid_input:
          FLAGS.valid_hour = os.path.basename(os.path.dirname(FLAGS.valid_input))
        else:
          FLAGS.valid_hour = None

      if not FLAGS.loop_train:
        FLAGS.valid_hour = None

      if FLAGS.valid_hour:
        os.system(f'mkdir -p {FLAGS.model_dir}/infos/{FLAGS.valid_hour}')
        if filename:
          ofile = f'{FLAGS.model_dir}/infos/{FLAGS.valid_hour}/{filename}.csv'
        else:
          ofile = f'{FLAGS.model_dir}/infos/{FLAGS.valid_hour}/{FLAGS.valid_out_file}'
      else:
        os.system(f'mkdir -p {FLAGS.model_dir}')
        ofile = f'{FLAGS.model_dir}/{filename}.csv' if filename else f'{FLAGS.model_dir}/{FLAGS.valid_out_file}'
 
      print_fn = logging.debug if FLAGS.work_mode == 'train' else logging.info
      print_fn(f'write valid result to {ofile}')
      if not self.write_fn:
        # TODO custome write_valid
        preds = self.preds
        m = {}
        m['label'] = self.y
        if isinstance(preds, dict):
          m.update(preds)
        else:
          m['pred'] = list(preds)
        m.update(self.x)
        m.update(self.other)
        # TODO 如果是 [bs, 2] 类似这样的array如何加入pandas
        for key in m:
          m[key] = list(m[key])
        df = pd.DataFrame(m)
        try:
          df.id = df.id.astype(int)
        except Exception:
          pass
        if 'id' in m:
          df = df.sort_values(['id'])
        df.to_csv(ofile, index=False, encoding="utf_8_sig")
        # if FLAGS.fold is None:
        #   df.to_csv(ofile2, index=False, encoding="utf_8_sig")
      else:
        write_args = inspect.getfullargspec(self.write_fn).args 
        kwargs = {}
        if 'others' in write_args:
          kwargs['others'] = self.other
        elif 'other' in write_args:
          kwargs['other'] = self.other
        self.write_fn(self.x, self.y, self.preds, ofile, **kwargs)

      self.write_valid_steps.add(self.eval_step)

  def on_train_begin(self, logs={}):
    if FLAGS.ev_first or FLAGS.work_mode == 'valid':
      if FLAGS.ev_first:
        if self.eval_step == 0:
          self.eval_step -= 1
      self.eval()
    # if FLAGS.first_interval_epoch > 0 and not FLAGS.ev_first:
    #   if self.eval_step == 0:
    #     self.eval_step -= 1
    if FLAGS.work_mode == 'valid':
      melt.save_eval_step()
      exit(0)
    return

  def on_train_end(self, logs={}):
    # if FLAGS.do_valid and FLAGS.metric_eval:
    #   if FLAGS.write_valid_final:
    #     self.write_valid()
    return

  # Need init iter here for we also do validation using validation_steps=num_valid_examples when train epoch end
  def on_epoch_begin(self, epoch, logs={}): 
    return

  def on_epoch_end(self, epoch, logs={}):
    # 即使是valid_interval_epochs == 1 也已经在on_batch_end处理了
    pass
    # if FLAGS.do_valid and FLAGS.metric_eval:
    #   self.eval()

  def _is_eval_step(self, step):
    return step % self.steps_per_eval == 0 or step == self.steps_per_epoch * FLAGS.num_epochs

  def is_eval_step(self, step):
    pre_step = step - FLAGS.steps_per_execution
    for step_ in reversed(range(pre_step + 1, step + 1)):
      if self._is_eval_step(step_):
        return step_
    return None

  def on_batch_end(self, batch, logs={}):
    self.step += FLAGS.steps_per_execution
    
    if FLAGS.do_valid and FLAGS.metric_eval:
      if FLAGS.metric_eval_interval_steps:
        if self.step % FLAGS.metric_eval_interval_steps == 0:
          self.eval()

      if FLAGS.first_interval_epoch > 0:
        if self.step == int(FLAGS.first_interval_epoch * self.steps_per_epoch):
          self.eval(silent=True)

      if FLAGS.second_interval_epoch > 0:
        if self.step == int(FLAGS.second_interval_epoch * self.steps_per_epoch):
          self.eval()

      if FLAGS.valid_interval_epochs and FLAGS.valid_interval_epochs > 0:
        # steps interval per eval
        step = self.is_eval_step(self.step)
        if step is not None:
          is_last = False
          # self.epoch = self.step / self.steps_per_epoch
          self.epoch = float(math.ceil(step / self.steps_per_eval)) * FLAGS.valid_interval_epochs
          # logging.debug('eval_on_batch_end', 'step', step, 'epoch', self.epoch, FLAGS.epochs, self.epoch >= FLAGS.epochs)
          if self.epoch >= FLAGS.epochs:
            is_last = True 
          if FLAGS.loop_train and not (FLAGS.round == FLAGS.num_rounds - 1):
            is_last = False
          
          self.is_last = is_last
          if not is_last:
            self.eval(logs=logs)
          else:
            if FLAGS.full_validation_final and FLAGS.num_valid:
              logging.debug('full_validation_final:', FLAGS.full_validation_final, 'num_valid before full valid:', FLAGS.num_valid)
              self.eval(logs=logs)
            self.eval(is_last, logs=logs)

    return

  def on_batch_begin(self, batch, logs={}):
    return
