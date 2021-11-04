#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2019-09-04 17:31:00.630021
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
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import Sequence
try:
  import tensorflow_addons as tfa
except Exception:
  pass

try:
  import tensorflow_model_optimization as tfmot
except Exception:
  pass

try:
  import nni.compression.tensorflow
  from nni.compression.tensorflow import OneshotPruner
  from nni.compression.tensorflow import LevelPruner
  from nni.compression.tensorflow import FPGMPruner
except Exception:
  pass

from collections import defaultdict
import inspect
import numpy as np
from tqdm import tqdm
try:
  from wandb.keras import WandbCallback
  HAS_WANDB = True
except Exception as e:
  print(e)
  HAS_WANDB = False

import gezi 
logging = gezi.logging

from husky.callbacks import *
from husky import optimization
from husky.callbacks.tqdm_progress_bar import TQDMProgressBar
from husky.ema import ExponentialMovingAverage

def setup_metrics(metrics):
  for i in range(len(metrics)):
    if metrics[i] == 'auc':
      metrics[i] = tf.keras.metrics.AUC()

# iter_ = iter
def train(model, 
          loss_fn,
          info,
          Dataset=None,  
          dataset=None,
          valid_dataset=None,
          valid_dataset2=None,
          test_dataset=None,
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          eval_keys=[],
          write_valid=None,
          valid_names=None,
          infer_names=None,
          infer_debug_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid',
          infer_suffix='.infer',
          write_streaming=False,
          optimizer=None,
          variables_list_fn=None,
          lr_params=None,
          init_fn=None,
          weights=1.0, 
          sep=',',
          out_hook=None,
          out_keys=[],
          callbacks=[],
          metrics=[],
          initial_epoch=0,
          return_info=False,
          dry_run=False):
  # assert isinstance(model, melt.Model), 'For keras use melt.Model instead of keras.Model'
  ## functional api keras model
  epochs = int(FLAGS.num_epochs)
  initial_epoch = initial_epoch or FLAGS.start_epoch
 
  strategy = melt.distributed.get_strategy()
  with strategy.scope():
    dataset = info['dataset']
    eval_dataset = info['eval_dataset']
    valid_dataset = info['valid_dataset']
    test_dataset = info['test_dataset']
  
    if not isinstance(model, melt.Model):
      FLAGS.use_info_dataset = True

    eval_info_dataset = info['eval_info_dataset'] if FLAGS.use_info_dataset else None
    test_info_dataset = info['test_info_dataset'] if FLAGS.use_info_dataset else None

    if not isinstance(model, melt.Model) or FLAGS.predict_on_batch:
      eval_info_dataset = eval_info_dataset or eval_dataset
      test_info_dataset = test_info_dataset or test_dataset

    # print('---------dataset', dataset)

    # TODO test for num_train change will learning rate decay ok ?
    steps_per_epoch = info['num_steps_per_epoch']
    steps_per_epoch_ = gezi.get('num_steps_per_epoch') or steps_per_epoch
    valid_steps_per_epoch = info['num_valid_steps_per_epoch']
    test_steps_per_epoch = info['num_test_steps_per_epoch']
    epochs_ = gezi.get('epochs') or epochs

    logging.debug('steps_per_epoch_', steps_per_epoch_, 'steps_per_epoch', steps_per_epoch)
    total_steps = epochs_ * steps_per_epoch_
    gezi.set('total_steps', total_steps)
    logging.debug('total_steps', total_steps)

    if FLAGS.buckets:
      FLAGS.recount_train_steps = True
      # 用buckets之后 必须 --keras_loop=0 而两次访问valid 顺序不一致了 有了shuffle ... 或者就是valid test 
      # 和train不同 不采用buckets valid 本身占用显存小 或者再适度设置小一点 --eval_batch_size
      # FLAGS.keras_loop = False
      # # steps_per_epoch = int(steps_per_epoch * 1.2)
      # # with strategy.scope():
      # if eval_dataset:
      #   step = 0
      #   for item in tqdm(eval_dataset, total=valid_steps_per_epoch, ascii=True):
      #     step += 1
      #   logging.info('valid_step1', valid_steps_per_epoch, 'valid_step2', step, step / valid_steps_per_epoch)
      #   valid_steps_per_epoch = step
      # if test_dataset:
      #   step = 0
      #   for item in tqdm(test_dataset, total=test_steps_per_epoch, ascii=True):
      #     step += 1
      #   logging.info('test_step1', test_steps_per_epoch, 'test_step2', step, step / test_steps_per_epoch) 
      #   test_steps_per_epoch = step

    if FLAGS.recount_train_steps:
      step = 0
      for item in tqdm(dataset, total=steps_per_epoch, desc='recount-train_steps', ascii=True):
        step += 1

      logging.info('step', steps_per_epoch, 'step2', step, step / steps_per_epoch)
      steps_per_epoch = step

    if isinstance(weights, str):
      logging.warning('input weights str not support in keras, please deal weight inside loss_fn')

    # --------------   optimzier
    def get_epoch_lrfn():
      # TODO move lr to other file
      LR_START = FLAGS.min_learning_rate
      # LR_MAX = 0.00005 * strategy.num_replicas_in_sync
      LR_MAX = FLAGS.learning_rate
      LR_MIN = FLAGS.min_learning_rate
      LR_RAMPUP_EPOCHS = int(FLAGS.warmup_epochs or FLAGS.warmup_proportion * FLAGS.num_epochs)  # 0.1 * epochs or just first 5 epochs
      assert LR_RAMPUP_EPOCHS > 0, f'warmup{FLAGS.warmup_proportion} * epochs {FLAGS.num_epochs}'
      LR_SUSTAIN_EPOCHS = FLAGS.sustain_epochs # 0
      LR_EXP_DECAY = FLAGS.epoch_lr_exp_decay # .8

      # https://www.kaggle.com/goldenlock/custom-training-loop-with-100-flowers-on-tpu/edit?rvi=1
      @tf.function
      def epoch_lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
          lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
          lr = LR_MAX
        else:
          if FLAGS.epoch_lr_decay_strategy == 0:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
          else:
            lr = (LR_MAX - LR_MIN) * (1. - epoch / FLAGS.num_epochs) ** LR_EXP_DECAY + LR_MIN
        return lr

      # Instiate optimizer with learning rate schedule
      class EpochLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return epoch_lrfn(epoch=step//steps_per_epoch_)

      return EpochLRSchedule()

    if not FLAGS.swa_start_epoch or FLAGS.opt_swa:
      swa_start_epoch = epochs_
    else:
      swa_start_epoch = FLAGS.swa_start_epoch
    if FLAGS.opt_swa:
      gezi.set('swa_start_step', int(FLAGS.swa_start_epoch * steps_per_epoch_))
      gezi.set('swa_steps', int(FLAGS.swa_freq * steps_per_epoch_))
    swa_epochs = epochs_ - swa_start_epoch
    logging.debug('swa_sgart_epohc', swa_start_epoch, 'swa_epochs', swa_epochs)
    total_steps2 = int((epochs_ - swa_epochs) * steps_per_epoch_)
    decay_steps = total_steps2 if (FLAGS.num_decay_epochs is None or FLAGS.num_decay_epochs <= 0) else int(FLAGS.num_decay_epochs * steps_per_epoch_)
    num_warmup_steps = FLAGS.warmup_steps or int(FLAGS.warmup_epochs * steps_per_epoch_) or int(FLAGS.warmup_proportion * decay_steps)
    # ic(total_steps2, decay_steps, num_warmup_steps)
    def _create_lr_schedule(lr):
      if FLAGS.optimizer.startswith('epoch-'):
        return get_epoch_lrfn()
      cycle = FLAGS.num_decay_epochs is not None and FLAGS.num_decay_epochs > 0
      lr_schedule = optimization.create_lr_schedule(lr, decay_steps + 1, 
                  num_warmup_steps=num_warmup_steps, end_lr=FLAGS.min_learning_rate,
                  power=FLAGS.lr_decay_power, decay_method=FLAGS.lr_decay_method,
                  cycle=cycle)
      return lr_schedule
    
    lr_schedule = FLAGS.learning_rate
    lr_schedules = []

    if optimizer is None:
      if gezi.get('optimizer'): # 默认会复用optimzier如果需要重置设置 gezi.set('optimizer', None)
        optimizer = gezi.get('optimizer')
        if FLAGS.optimizer.startswith('bert-') or FLAGS.optimizer.startswith('schedule-') or FLAGS.optimizer.startswith('epoch-'):
          lr_schedule = _create_lr_schedule(FLAGS.learning_rate)
        optimizers = gezi.get('optimizers', [optimizer])
      else:
        optimizers = []
        kwargs = {
            'momentum': FLAGS.opt_momentum,
            'nesterov': FLAGS.opt_nesterov,
        }
        
        # TODO 当前只支持多个相同类型optimizer！
        if FLAGS.optimizer.startswith('bert-') or FLAGS.optimizer.startswith('schedule-') or FLAGS.optimizer.startswith('epoch-'):
          for lr in FLAGS.lrs:
            lr_schedule = _create_lr_schedule(lr)
            lr_schedules.append(lr_schedule)
            opt_type = FLAGS.optimizer.split('-')[-1]
            optimizer = optimization.create_optimizer(
                  lr_schedule = lr_schedule,
                  epsilon=FLAGS.opt_epsilon,
                  weight_decay=FLAGS.opt_weight_decay, 
                  optimizer_type=opt_type,
                  **kwargs)
            optimizers.append(optimizer)
        else:
          opt_name = FLAGS.optimizer
          names = {
                  'adam': 'Adam', 
                  'sgd': 'SGD',
                  'lazyadam': 'LazyAdam',
                  }
          if opt_name in names:
            opt_name = names[opt_name]

          try:
            Opt = getattr(tf.keras.optimizers, opt_name)
          except Exception:
            import tensorflow_addons as tfa
            Opt = getattr(tfa.optimizers, opt_name)
            
          args = inspect.getargspec(Opt).args    
          kwargs_ ={}
          if 'epsilon' in args:
            kwargs_['epsilon'] = FLAGS.opt_epsilon
          for arg in kwargs:
            if arg in args:
              kwargs_[arg] = kwargs[arg]
          
          for lr in FLAGS.lrs:
            lr_schedules.append(lr)
            optimizer = Opt(lr=lr, **kwargs_)
            optimizers.append(optimizer)
          
        optimizer_ = optimizer
        if len(optimizers) > 1:
          import tensorflow_addons as tfa
          opt_layers = gezi.get('opt_layers')
          assert opt_layers, 'For multiple optimizers must set manualy set multiple opt_layers'
          assert len(opt_layers) == len(optimizers)
          logging.debug('opt_layers:', opt_layers)
          optimizers_and_layers = list(zip(optimizers, opt_layers))
          optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
          gezi.set('optimizers', optimizers)
          
    logging.info('round:', FLAGS.round, 'loss_fn:', loss_fn)
    if len(optimizers) == 1:
      try:
        logging.info('optimizer:', optimizer, 'lr:', optimizer.lr, 'init_lr:', FLAGS.learning_rate)
      except Exception:
        logging.info('optimizer:', optimizer, 'lr:', optimizer.optimizer.lr, 'init_lr:', FLAGS.learning_rate)
    else:
      try:
        logging.info('optimizers[0]:', optimizers[0], 'lrs[0]:', optimizers[0].lr)
      except Exception:
        logging.info('optimizers[0]:', optimizers[0], 'lrs[0]:', optimizers[0].optimizer.lr)
      lrs = list(zip([x[0] for x in opt_layers], FLAGS.lrs))
      logging.info('init_lrs:', lrs)
              
    logging.info('total_steps:', total_steps, 'warmup_steps:', num_warmup_steps, 'end_lr:', FLAGS.min_learning_rate)
    gezi.set('optimizer', optimizer)
    if FLAGS.reset_lr:
      if len(optimizers) == 1:
        optimizer._set_hyper('learning_rate', lr_schedule)
      else:
        for optimizer, lr_schedule in zip(optimizers, lr_schedules):
          optimizer._set_hyper('learning_rate', lr_schedule)

    #------------------------setup callbacks
    # if not gezi.get('callbacks'):
    # checkpoint_dir = os.path.join(FLAGS.model_dir, 'ckpt')
    # os.system(f'mkdir -p {checkpoint_dir}')
    # checkpoint_path =  os.path.join(checkpoint_dir, 'model.ckpt-{epoch:02d}')
    # save_callback = keras.callbacks.ModelCheckpoint(
    # checkpoint_path, save_weights_only=True,
    # period=max(FLAGS.save_interval_epochs, 1),
    # verbose=int(FLAGS.debug))
    pretrained_dir = FLAGS.pretrained_dir if not FLAGS.pretrained_dir or os.path.exists(FLAGS.pretrained_dir) else os.path.join(os.path.dirname(FLAGS.model_dir), FLAGS.pretrained_dir)
    iterator = iter(dataset)
    ckpt_callback = CheckpointCallback(FLAGS.model_dir, model, optimizer, iterator=iterator, pretrained_dir=pretrained_dir)

    setup_metrics(FLAGS.metrics)
    metrics = metrics or []
    metrics = metrics + FLAGS.metrics       
    logging.debug('metrics:', metrics)

    if not FLAGS.train_scratch and not gezi.get('model_loaded'):
      ckpt_callback.on_train_begin_()
      model = ckpt_callback.model # might changed to new model from tf.kears.models.load_model return
      gezi.set('model_loaded', True)
              
    if FLAGS.reset_lr:
      optimizer._set_hyper('learning_rate', lr_schedule)
    # must after load optimizer
    if FLAGS.reset_global_step:
      optimizer.iterations.assign(0)  # (global step not reset to 0) otherwise lr scheduler will fail lr == 0 
      model.step = 0
      gezi.set('total_step', 0)
      gezi.try_remove(f'{FLAGS.model_dir}/total_step.txt')
      gezi.try_remove(f'{FLAGS.model_dir}/model_step.txt')
    else:
      # ic(optimizer.iterations, optimizer.iterations / steps_per_epoch, steps_per_epoch)
      iterations = optimizer.iterations.numpy() if hasattr(optimizer.iterations, 'numpy') else optimizer.iterations
      if not iterations:
        model_step = gezi.read_int_from(f'{FLAGS.model_dir}/model_step.txt', 0)
        optimizer.iterations.assign(model_step)
      iterations = optimizer.iterations.numpy() if hasattr(optimizer.iterations, 'numpy') else optimizer.iterations
      if initial_epoch:
        optimizer.iterations.assign(int(initial_epoch * steps_per_epoch))
      elif iterations > 0:
        initial_epoch = int(optimizer.iterations / steps_per_epoch)

      # 如果从optimizer获取到step 则不再依赖文件 total_step.txt存储的step信息
      if optimizer.iterations != 0: 
        ## 注意不要下面这样写入total_step.txt 
        # melt.set_total_step(optimizer.iterations.numpy())
        gezi.set('total_step', optimizer.iterations.numpy())
        
    gezi.set('global_step', optimizer.iterations)
    # inputs = gezi.list_files(FLAGS.train_input)
    # example = next(iter_(dataset or Dataset('train').make_batch(FLAGS.batch_size, [inputs[0]])))[0]
    # model(example)
    # ic(model.step, gezi.get('global_step'), gezi.get('total_step'))
    kwargs = {}
    
    if tf.__version__ >= '2.3':
      kwargs = {
        'experimental_steps_per_execution': FLAGS.steps_per_execution,
      }
    else:
      # logging.warning('Not support experimental_steps_per_execution with tf < 2.3 reset FLAGS.steps_per_execution to 1')
      FLAGS.steps_per_execution = 1

    model.my_loss = loss_fn
    model.my_optimizer = optimizer
    # model.my_metrics = metrics  # 奇怪 SemanticSeg [0.0533 0.0403 0.1423 0.123 ] 如果设置metrics返回的wrapper那个就不对了。。。 TODO

    if FLAGS.final_sparsity < 1:
      # TODO not work..
      # ValueError: Please initialize `Prune` with a supported layer. Layers should either be a `PrunableLayer` instance, or should be supported by the PruneRegistry.
      # You passed: <class 'tensorflow.python.keras.engine.functional.Functional'>
      prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
      pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=FLAGS.initial_sparsity,
                                                                     final_sparsity=FLAGS.final_sparsity,
                                                                     begin_step=0,
                                                                     end_step=total_steps)
      }
      model = prune_low_magnitude(model.get_model(), **pruning_params)

      # prune_config = {
      #   'level': {
      #       'model_name': FLAGS.pruner_name,
      #       'pruner_class': nni.compression.tensorflow.LevelPruner,
      #       'config_list': [{
      #           'sparsity': FLAGS.final_sparsity,
      #           'op_types': ['default'],
      #       }]
      #   },
      # }
      # def create_pruner(model, pruner_name):
      #   pruner_class = prune_config[pruner_name]['pruner_class']
      #   config_list = prune_config[pruner_name]['config_list']
      #   return pruner_class(model, config_list)

      # pruner = create_pruner(model, 'level')

      # configure_list = [{
      #   'sparsity': FLAGS.final_sparsity,
      #   'op_types': ['default'],
      # }]

      # pruner = OneshotPruner(model, configure_list)
      # model = pruner.compress()

      FLAGS.run_eagerly = True

    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=FLAGS.run_eagerly,
        **kwargs,
      )

    gezi.set('model', model)
    gezi.set('model_compiled', True)

    # best_checkpoint_path = os.path.join(FLAGS.model_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5')
    # save_best = keras.callbacks.ModelCheckpoint(best_checkpoint_path, verbose=1, save_best_only=True)

    valid_loss_callback, eval_callback, test_callback = None, None, None

    if not FLAGS.train_only:
      valid_loss_callback = ValidLossCallback(valid_dataset, loss_fn=loss_fn) 
      eval_callback = EvalCallback(model, eval_dataset, eval_fn, eval_info_dataset, 
                                  steps=valid_steps_per_epoch,
                                  num_examples=info['num_valid_examples'],  
                                  steps_per_epoch=steps_per_epoch,
                                  write_valid=write_valid, 
                                  write_fn=valid_write_fn,
                                  loss_fn=loss_fn,
                                  pretrained_dir=pretrained_dir)             
      if test_dataset is not None:             
        test_callback = TestCallback(model, test_dataset, test_info_dataset, test_steps_per_epoch, 
                                     num_examples=info['num_test_examples'], ofile=FLAGS.test_out_file,
                                     write_fn=infer_write_fn, inference_fn=inference_fn)
    
    nbatch_logger_callback = NBatchLoggerCallback(FLAGS.valid_interval_steps)

    gezi.set('eval_callback', eval_callback)

    verbose = FLAGS.keras_verbose
    callbacks_ = [
                ## UtilsCallback(),
                ## save_best,
                AccGradientsCallback(),
                valid_loss_callback, 
                nbatch_logger_callback,
                TimerCallback(),
                L2LossCallback(),
                ckpt_callback,
                eval_callback,
                test_callback,
              ]

    # 使用自定义tqdm进度条取代keras默认
    if FLAGS.keras_custom_progress_bar:
      tqdm_callback = TQDMProgressBar(steps_per_execution=FLAGS.steps_per_execution,
                                      leave_epoch_progress=FLAGS.leave_epoch_progress,
                                      leave_overall_progress=FLAGS.leave_overall_progress,
                                      show_epoch_progress=FLAGS.show_epoch_progress,
                                      show_overall_progress=FLAGS.show_overall_progress,
                                      update_each_epoch=FLAGS.update_each_epoch,
                                      initial_epoch=int(initial_epoch))
      # ic(tqdm_callback, initial_epoch, FLAGS.steps_per_execution)
      callbacks_ += [tqdm_callback]
      verbose = 0
    else:
      verbose = FLAGS.keras_verbose

    # colab tpu TensorBoard也需要设置gcs path写 local路径不行 
    # TODO 加入下面 profile_batch速度也慢很多... 只是debug用暂时
    if FLAGS.keras_tensorboard:
      log_dir = FLAGS.log_dir
      if gezi.get('tpu'):
        assert FLAGS.gcs_dir, 'for tpu need gcs dir'
        log_dir = FLAGS.gcs_dir
      start = 2
      end = FLAGS.profile_interval_steps or 3
      profile_batch= (start, end)
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=profile_batch, write_graph=FLAGS.write_graph)
      callbacks_ += [tensorboard_callback]

    if FLAGS.final_sparsity < 1:
      callbacks_ += [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=FLAGS.logdir),
      ]

    if swa_epochs:
      swa = SWA(
          start_epoch=swa_start_epoch, 
          init_lr=FLAGS.learning_rate,
          lr_schedule='cyclic', 
          swa_lr=FLAGS.min_learning_rate,
          swa_lr2=FLAGS.learning_rate * FLAGS.swa_lr_ratio,
          swa_freq=FLAGS.swa_freq,
          swa_warmup=FLAGS.swa_warmup,
          model_path=f'{FLAGS.model_dir}/model.h5',
          verbose=1,
          fake_run=FLAGS.swa_fake_run)
      gezi.set('swa', swa)
      callbacks_ = [swa, *callbacks_]
  
    # if HAS_WANDB and FLAGS.wandb_key:
    #   callbacks_ += [WandbCallback(save_model=False)]
    callbacks = callbacks_ + callbacks
    callbacks = [x for x in callbacks if isinstance(x, Callback)]

    gezi.set('callbacks', callbacks)
    # ic(callbacks)

    logger = None
    if FLAGS.log_dir and FLAGS.write_summary and FLAGS.write_metric_summary:
      logger = melt.get_summary_writer()

    for callback in callbacks:
      callback.set_model(model)
      callback.steps_per_epoch = steps_per_epoch
      callback.valid_steps_per_epoch = valid_steps_per_epoch
      callback.num_valid_examples = info['num_valid_examples']
      callback.num_test_examples = info['num_test_examples']
      callback.logger = logger
      callback.optimizer = optimizer

    info['optimizer'] = optimizer
    info['model'] = model
    info['loss_fn'] = loss_fn
    info['metrics'] = metrics
    info['callbacks'] = callbacks
    info['eval_callback'] = eval_callback
    info['test_callback'] = test_callback
    info['compiled_metrics'] = model.compiled_metrics
    info['history'] = defaultdict(list)
    
    gezi.set('info', info)
    gezi.set('history', info['history'])

    if not tf.executing_eagerly() and tf.__version__ < '2':
      sess = melt.get_session()
      init_op = tf.group(tf.compat.v1.global_variables_initializer(), #variables_initializer(global_variables())
                        tf.compat.v1.local_variables_initializer()) #variables_initializer(local_variables())
      sess.run(tf.compat.v1.global_variables_initializer())
      K.set_session(sess)
      logging.debug('tf.trainable_variables:', tf.compat.v1.trainable_variables())

      def _init_iter(iter, subset, index=0):
        if iter is not None and hasattr(iter, 'initializer'):
          need_feed = FLAGS.train_loop and FLAGS.rounds > 1 and not tf.executing_eagerly() and FLAGS.feed_dataset
          if not need_feed:
            sess.run(iter.initializer)
          else:
            sess.run(iter.initializer, feed_dict={gezi.get_global(f'{subset}_{index}'): melt.Dataset.get_filenames_(subset)})

      iter_ = gezi.get_global('iter')
      _init_iter(iter_, 'train')
      valid_iter = gezi.get_global('valid_iter', None)
      _init_iter(valid_iter, 'valid')
      valid_iter2 = gezi.get_global('valid_iter2', None)
      _init_iter(valid_iter2, 'valid', 1)
      test_iter = gezi.get_global('test_iter', None)
      _init_iter(test_iter, 'test')

    if hasattr(model, 'init'):
      model.init()

    if hasattr(model, 'restore'):
      model.restore() 

    history = {}
    if FLAGS.cv_valid_only or FLAGS.cv_test_only:
      weights_file = f'{FLAGS.model_dir}/cv/model_{FLAGS.fold}.h5'
      logging.info(f'load_weights from {weights_file}')
      model.load_weights(weights_file)
      if not FLAGS.cv_test_only:
        eval_callback.eval()
      if (FLAGS.do_test and FLAGS.test_interval_epochs >= 0) or FLAGS.cv_test_only:
        test_callback.test()
    elif FLAGS.steps < 0:
      ckpt_callback.save(is_last=True)
    elif FLAGS.work_mode == 'valid' or FLAGS.mode == 'valid':
      melt.print_model(model, depth=FLAGS.print_depth)
      eval_callback.eval(is_last=True if FLAGS.is_last_eval is None else FLAGS.is_last_eval)
    elif FLAGS.work_mode == 'test' or FLAGS.mode == 'test':
      melt.print_model(model, depth=FLAGS.print_depth)
      test_callback.test()
    else:
      # main training loop
      K.set_learning_phase(1)
      model.mode = 'train'
      if not dry_run:
        if FLAGS.ema_inject:
          ema = ExponentialMovingAverage(model, momentum=FLAGS.opt_ema_momentum) # 在模型compile之后执行
          gezi.set('ema', ema)
          ema.inject() # 在模型compile之后执行

        if FLAGS.num_epochs2:
          epochs = FLAGS.num_epochs2

        # TODO 
        if FLAGS.custom_loop:
          raise NotImplementedError
          # projects/ai/qqbrowser/baseline/tensorflow/train.py
          @tf.function
          def train_step(x, y):
              with tf.GradientTape() as tape:
                  predictions, _ = model(x, training=True)
                  loss = loss_object(y, predictions)
              gradients = tape.gradient(loss, model.get_variables())
              model.optimize(gradients)
              
          for epoch in tqdm(epochs):
            for x, y in tqdm(dataset, total=steps_per_epoch):
              train_step(x, y)
        else:
          history = model.fit(dataset,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=valid_dataset if FLAGS.keras_validation else None,
                              validation_steps=valid_steps_per_epoch if not FLAGS.keras_validation_steps else FLAGS.keras_validation_steps,
                              callbacks=callbacks,
                              initial_epoch=initial_epoch,
                              sample_weight=gezi.get('sample_weight'),
                              class_weight=gezi.get('class_weight'),
                              verbose=verbose,
                            )
        melt.inc_total_step(total_steps)
      else:
        logging.info('keras mode dry_run done')

      K.set_learning_phase(0)
      gezi.set('keras/history', history)
      if HAS_WANDB and FLAGS.wandb:
        try:
          wandb.log({'model_loss': wandb.Image(gezi.plot.model_loss(), caption='model_loss')})
        except Exception:
          pass
      
  return history
