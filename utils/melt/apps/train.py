#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-17 10:30:20.286494
#   \Description   METRIC=1 python ....    just run metric eval
# ==============================================================================

"""
not supporting averaging and multi gpu yet  @TODO
 [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)

 here what we do is 
 create train_op from loss
 and may be using multi gpu deal
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__pacage__ = None 

import sys 
import os 
import inspect
import traceback
import time

import gezi
from gezi import logging
import melt  

from tensorflow.python import debug as tf_debug
import tensorflow as tf
if tf.__version__ < '2':
  ## TODO for tf2 not ok even with tf_slim
  #import tensorflow.contrib.slim as slim
  import tf_slim as slim
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline

# from tqdm.notebook import tqdm
from gezi import tqdm
import numpy as np
import math
import pandas as pd
import copy

from multiprocessing import Manager
try:
  import pymp
except Exception:
  pass

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

# This is dangerous.. as from tensorflow.keras import backend as K will use different code(tf code) compare to from keras import backend as K(keras.code)
from tensorflow import keras
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# tfe = tf.contrib.eager

if sys.version_info > (3,):
  long = int

from melt.apps.config import *
from melt.apps.init import is_inited
from melt.flow.train_once import profile_step

def get_global_scope():
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  return global_scope

def gen_learning_rate(init_lr, num_steps_per_epoch=None, index=0):
  #TODO if app crash reload then we should set smaller learning rate, may adgrad can combine with exponential_decay ?
  #copy from im2txt\im2txt\train.py
  _learning_rate_decay_fn = None
  logging.debug('---inital learning rate:', init_lr)
  with tf.compat.v1.variable_scope(name_or_scope='', reuse=tf.compat.v1.AUTO_REUSE):  
    #TODO righ now not to pass float learning_rate as will be variable in optimizer.py and save
    #if restore it will get learning rate form checkpoint, even if you give another learning rate 
    #this might be confusing, so just make it constant, if restart training you can change learning rate 
    #remeber if using decay by defualt beaviour will restore global_step from checkpoint
    #so you can restart and get decayed learning rate direclty, restart is same as training without start
    #you can also set smaller learning rate for example if learning_rate 0.1 before then decay learning rate is 
    #0.1 * decay you can set learning rate to 0.001 manualy after when restart training, then it means you start
    #with 0.001 * decay (0.01 * decayed_learning_rate)
    lr_name = 'learning_rate' if not index else 'learning_rate%d' % index
    if not FLAGS.dynamic_learning_rate:
      if FLAGS.learning_rate_decay_factor > 0 or FLAGS.learning_rate_method != 'decay':
        learning_rate = tf.constant(init_lr)
        #learning_rate = FLAGS.learning_rate
      else:
        learning_rate = tf.compat.v1.get_variable(
          lr_name, [],
          trainable=False,
          initializer=tf.compat.v1.constant_initializer(init_lr),
          collections=[])
    else:
      logging.debug('using dyanmic learning rate')
      learning_rate = tf.compat.v1.get_variable(
        lr_name, [],
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(init_lr))

      if FLAGS.learning_rate_patience:
        assert FLAGS.learning_rate_decay_factor > 0
        logging.debug('adjust learning rate by patience {} and decay_factor *{}'.format(FLAGS.learning_rate_patience, FLAGS.learning_rate_decay_factor))
        return learning_rate, None
  
    #if not init_lr > 0:
    #  assert FLAGS.learning_rate_values, 'if learning rate is 0 then must set learnint rate values'

    if FLAGS.learning_rate_values:
      if not learning_rate_:
        learning_rate_values = [float(lr) for lr in FLAGS.learning_rate_values.split(',')]
        assert FLAGS.learning_rate_step_boundaries or FLAGS.learning_rate_epoch_boundaries 
        if FLAGS.learning_rate_step_boundaries:
          assert FLAGS.learning_rate_epoch_boundaries is None, 'use step boundaries or epoch boundaries?'
          boundaries = [long(bound) for bound in FLAGS.learning_rate_step_boundaries.split(',')]
        else:
          assert num_steps_per_epoch is not None, 'need epoch info if using epoch boundaries'
          boundaries = [long(float(epoch_bound) * num_steps_per_epoch) for epoch_bound in FLAGS.learning_rate_epoch_boundaries.split(',')]
        
        assert len(learning_rate_values) == len(boundaries) + 1, \
          'len_values:{} len_bouddaries:{}'.format(len(learning_rate_values), len(boundaries))

        logging.debug('learning rate values:{}, epoch_bounds:{} boundaries:{}'.format(
            FLAGS.learning_rate_values, FLAGS.learning_rate_epoch_boundaries, ','.join(map(str, boundaries))))

        def _learning_rate_decay_fn(learning_rate, global_step):
          # return tf.train.piecewise_constant(
          return melt.train.piecewise_constant(
            global_step,
            boundaries, 
            learning_rate_values)
    elif FLAGS.learning_rate_decay_factor > 0:
      #assert FLAGS.learning_rate_values is None, 'use exponential_decay or piecewise_constant?'
      #NOTICE if you do finetune or other things which might change batch_size then you'd better direclty set num_steps_per_decay
      #since global step / decay_steps will not be correct epoch as num_steps per epoch changed
      #so if if you change batch set you have to reset global step as fixed step
      assert FLAGS.num_steps_per_decay or (FLAGS.num_epochs_per_decay and num_steps_per_epoch), 'must set num_steps_per_epoch or num_epochs_per_decay and num_steps_per_epoch'
      decay_steps = FLAGS.num_steps_per_decay or int(num_steps_per_epoch * FLAGS.num_epochs_per_decay)    
      decay_start_step = FLAGS.decay_start_step or int(num_steps_per_epoch * FLAGS.decay_start_epoch)
      # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
      logging.debug('learning_rate_decay_factor:{} decay_epochs:{} decay_steps:{} decay_start_epoch:{} decay_start_step:{}'.format(
          FLAGS.learning_rate_decay_factor, FLAGS.num_epochs_per_decay, decay_steps, FLAGS.decay_start_epoch, decay_start_step))

      def _learning_rate_decay_fn(learning_rate, global_step):
        return melt.train.exponential_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            decay_rate=FLAGS.learning_rate_decay_factor,
            decay_start_step=decay_start_step,
            staircase=True)
    elif FLAGS.learning_rate_method == 'cosine':
      assert FLAGS.num_steps_per_decay or (FLAGS.num_epochs_per_decay and num_steps_per_epoch), 'must set num_steps_per_epoch or num_epochs_per_decay and num_steps_per_epoch'
      decay_steps = FLAGS.num_steps_per_decay or int(num_steps_per_epoch * FLAGS.num_epochs_per_decay)    
      logging.debug('learning_rate_decay_factor:{} decay_epochs:{} decay_steps:{}'.format(
          FLAGS.learning_rate_decay_factor, FLAGS.num_epochs_per_decay, decay_steps))
      def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.compat.v1.train.cosine_decay_restarts(
            learning_rate,
            global_step,
            first_decay_steps=decay_steps,
            t_mul=FLAGS.learning_rate_cosine_t_mul,
            m_mul=FLAGS.learning_rate_cosine_m_mul,
            alpha=FLAGS.learning_rate_cosine_alpha)
    else:
      # TODO support Slanted triangular learning rate
      # https://medium.com/@hiromi_suenaga/deep-learning-2-part-2-lesson-10-422d87c3340c 
      logging.debug('Will ignore FLAGS.learning_rate_values since you have learning rate not 0!')

    learning_rate_decay_fn = _learning_rate_decay_fn
    return learning_rate, learning_rate_decay_fn

def get_optimizer_byname(optimizer_name, learning_rate):
  if optimizer_name == 'momentum':
    momentum = FLAGS.opt_momentum if FLAGS.opt_momentum > 0 else FLAGS.momentum
    optimizer = lambda lr: tf.compat.v1.train.MomentumOptimizer(lr, momentum=momentum) 
  elif optimizer_name == 'adafactor':
    from tensor2tensor.utils import adafactor
    # let adafactor just using it's internal learning rate and default params
    # TODO FIXME sparse not support ...
    optimizer = adafactor.AdafactorOptimizer()
  elif optimizer_name == 'multistep':
    # even embedding set cpu still say resource try from device gpu to cpu WHY ?
    from tensor2tensor.utils import multistep_optimizer
    optimizer = multistep_optimizer.MultistepAdamOptimizer
  elif optimizer_name == 'yellowfin':
    # must set embedding on cpu , then can run(like adagrad adadelta) but work poorly
    from tensor2tensor.utils import yellowfin
    optimizer = yellowfin.YellowFinOptimizer
  else:
    optimizer_fn = melt.util.get_optimizer(optimizer_name)
    assert callable(optimizer_fn)
    if tf.__version__ < '2':
      kwargs = {'learning_rate': learning_rate}
    else:
      kwargs = {'learning_rate': learning_rate.numpy()}
    if 'wd' in inspect.getfullargspec(optimizer_fn).args:
      kwargs['wd'] = FLAGS.opt_weight_decay
    if 'weight_decay' in inspect.getfullargspec(optimizer_fn).args:
      kwargs['weight_decay'] = FLAGS.opt_weight_decay  
    if 'epsilon' in inspect.getfullargspec(optimizer_fn).args:
      kwargs['epsilon'] = FLAGS.opt_epsilon
      # notice on https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/opt/python/training/weight_decay_optimizers.py#L379-L412
      # need to do this(* learning_rate) for weight decay, bert adam's implementation do not need this

    logging.debug('optimizer args', kwargs)
    optimizer = optimizer_fn(**kwargs)
  return optimizer

def create_optimizer(optimizer, optimizer_name, init_lr, num_steps_per_epoch, global_step, num_epochs, finetune_start_step=0, index=0):
  if FLAGS.use_horovod:
    import horovod.tensorflow as hvd

  if not optimizer_name.startswith('bert'):
    learning_rate, learning_rate_decay_fn = gen_learning_rate(init_lr, num_steps_per_epoch, index)
    if learning_rate_decay_fn is not None:
      learning_rate = learning_rate_decay_fn(learning_rate, global_step - finetune_start_step)
  else:
    num_train_steps = int(
      num_steps_per_epoch * (FLAGS.num_decay_epochs or num_epochs))
    if FLAGS.warmup_steps:
      num_warmup_steps = FLAGS.warmup_steps 
    elif FLAGS.warmup_epochs:
      num_warmup_steps = num_steps_per_epoch * FLAGS.warmup_epochs
    elif FLAGS.warmup_proportion:
      num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion) 
    else:
      num_warmup_steps = 0

    if FLAGS.use_horovod:
      num_warmup_steps = max(int(num_warmup_steps / hvd.size()), 1)

    logging.debug('num_train_steps', num_train_steps, 'num_warmup_steps', num_warmup_steps, 'warmup_proportion', FLAGS.warmup_proportion)

    learning_rate = melt.training.bert.optimization.create_lr(
                      global_step, init_lr, num_train_steps + 1, num_warmup_steps, 
                      min_learning_rate=FLAGS.min_learning_rate)

  #do not let optimizer do decay again!
  learning_rate_decay_fn = None 

  if FLAGS.round == 0:
    if optimizer is None:
      if optimizer_name.startswith('bert'):
        num_train_steps = int(
          num_steps_per_epoch * (FLAGS.num_decay_epochs or num_epochs))
        if FLAGS.warmup_steps:
          num_warmup_steps = FLAGS.warmup_steps 
          if FLAGS.use_horovod:
            num_warmup_steps = max(int(num_warmup_steps / hvd.size()), 1)
        elif FLAGS.warmup_epochs:
          num_warmup_steps = num_steps_per_epoch * FLAGS.warmup_epochs
        elif FLAGS.warmup_proportion:
          num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion) 
        else:
          num_warmup_steps = 0
        logging.debug('num_train_steps', num_train_steps, 'num_warmup_steps', num_warmup_steps, 'warmup_proportion', FLAGS.warmup_proportion)
        
        if '-' in optimizer_name:
          optimizer = get_optimizer_byname(optimizer_name.split('-')[-1], learning_rate)
        else:
          optimizer = melt.training.bert.optimization.create_optimizer(
            learning_rate=learning_rate,
            weight_decay=FLAGS.opt_weight_decay,
            epsilon=FLAGS.opt_epsilon,
            min_learning_rate=FLAGS.min_learning_rate)
      else:
        optimizer = get_optimizer_byname(optimizer_name, learning_rate)

    logging.debug('optimizer:{} {}'.format(optimizer_name, optimizer))
    
  return optimizer, learning_rate, learning_rate_decay_fn

def train_flow(ops, 
               names=None, 
               gen_feed_dict_fn=None, 
               deal_results_fn=None, 
               eval_ops=None, 
               eval_names=None,
               gen_eval_feed_dict_fn=None, 
               deal_eval_results_fn=melt.print_results,
               optimizer=None, 
               learning_rate=0.1, 
               num_steps_per_epoch=None,
               model_dir=None, 
               log_dir=None,
               metric_eval_fn=None, 
               inference_fn=None,
               debug=False,
               summary_excls=None,
               init_fn=None,
               restore_fn=None,
               restore_include=None,
               restore_exclude=None,
               save_all_scope=False,
               variables_to_restore=None, 
               variables_to_save=None,
               output_collection_names=None, 
               output_node_names=None,
               num_steps=None,
               num_epochs=None,
               num_train_examples=None,
               model=None,
               variables_list_fn=None,  # used when optimizer is a list
               callbacks=[],
               sess=None):
  """
     #variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8']) not used much 
     variables_to_save might be used but will hack here since we also want to save like seq2seq/OptimizeLoss/
  """
  assert is_inited(), 'Forget to call melt.apps.init() before using melt.apps.train_flow?'
  if FLAGS.use_horovod:
    import horovod.tensorflow as hvd
  if sess is None:
    sess = melt.get_session()

  model_dir = model_dir or FLAGS.model_dir
  log_dir = log_dir or FLAGS.log_dir

  logging.debug('clip_gradients:{}'.format(FLAGS.clip_gradients))

  num_gpus = melt.num_gpus()
  
  #batch size right now not define here, but in app code like input_app.py
  
  #NOTICE since melt.__init__.py with from melt.flow import * then you can not 
  #use melt.flow.train.train_flow but you can always use
  #from melt.flow.train.train_flow import train_flow

  # Set up the training ops.
  #notice '' only works in tf >= 0.11, for 0.10 will always add OptimeizeLoss scope
  #the diff is 0.10 use variable_op_scope and 0.11 use variable_scope
  optimize_scope = None if FLAGS.optimize_has_scope else ''
  # # NOTICE! initialzer value is step get from model check point if exits otherwise 0
  # #will not get step value from checkpoint since you might change batch size it is safe to set step by epoch num and batch size
  # #this is controlled by melt.apps.flow where global_step var is removed from restore var list 
  # #if set num_steps_per_decay then inital step actually the same as readding from check point global step not modify for batch size change
  # #be realy golbal step(not fixed global step)
  # # TODO FIXME not flexible... since if you want to use global step in classifier graph.. can not tf.train.get_or_create_global_step()
  # initial_step = melt.get_global_step(model_dir, num_steps_per_epoch, fix_step=(not FLAGS.num_steps_per_decay)) if FLAGS.global_step is None else FLAGS.global_step
  # logging.info('global_step init with initial_step from model_dir as %d' % initial_step)
  # # TODO right now has global_scope above global_step might need to remove using function creator show_and_tell/global_step (DT_INT64) []
  # # global_step = tf.get_variable(tf.GraphKeys.GLOBAL_STEP, shape=[], dtype=tf.int64, 
  # #                               initializer=tf.constant_initializer(initial_step))  
  # # or can use get_variable(.. collections=['global_step']) but notice her is later then you build graph... 
  # # tf.add_to_collection('global_step', global_step)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  if FLAGS.round != 0:
    initial_step = 0
    sess.run(tf.assign(global_step, tf.constant(initial_step, dtype=tf.int64)))

  if FLAGS.use_finetune_step:
    # NOTICE unlike global step this one will be save to checkpoint and read out without any change 
    finetune_start_step = tf.compat.v1.get_variable('finetune_start_step', shape=[], dtype=tf.int64, 
                                           initializer=tf.compat.v1.constant_initializer(initial_step))
  elif FLAGS.use_finetune2_step:
    # NOTICE if 'finetune_start_step2' then will try to load finetune_start_step2 from checkpoint.. where there only fine_start_step..
    finetune_start_step = tf.compat.v1.get_variable('finetune2_start_step', shape=[], dtype=tf.int64, 
                                           initializer=tf.compat.v1.constant_initializer(initial_step))    
  else:
    finetune_start_step = 0

  logging.debug('num_steps_per_epoch:', num_steps_per_epoch)

  if variables_list_fn and callable(variables_list_fn):
    variables_list = variables_list_fn()
  else:
    variables_list = None

  optimizers = None if FLAGS.round == 0 else gezi.get_global('opts')
  opts = []
  optimizer_results = []
  learning_rates_ = []
  if FLAGS.optimizers or isinstance(optimizer, (list, tuple)):
    # multiple optimizers
    optimizer_names = FLAGS.optimizers.split(',')
    if optimizers is None:
      if isinstance(optimizer, (list, tuple)):
        optimizers = optimizer
      else: 
        if optimizer is not None:
          optimizers = optimizer
        else:
          optimizers = [None] * len(optimizer_names)
    if FLAGS.num_optimizers:
      optimizers = optimizers[:FLAGS.num_optimizers]
    assert variables_list and len(variables_list) == len(optimizers), f'{len(variables_list)} {len(optimizers)}'
    if not FLAGS.learning_rates:
      learning_rates = [FLAGS.learning_rate] * len(optimizers)
    else:
      learning_rates = [float(x) for x in FLAGS.learning_rates.split(',')]
    
    for i in range(len(optimizers)):
      index = i if FLAGS.learning_rates else 0
      opt, lr, lr_decay = create_optimizer(optimizers[i], optimizer_names[i], learning_rates[i], 
              num_steps_per_epoch + 1, global_step, num_epochs, finetune_start_step, index=index)
      optimizer_results.append([opt, lr, lr_decay, variables_list[i]])
      opts.append(opt)
      learning_rates_.append(lr)
  else:
    # single optimizer
    # if input is a truely optimzier then we only use create_optimzier for learnint rate HACK
    if optimizers is not None:
      optimizer = optimizers[0]
    # NOTICE for round > 0, optimizer is just reuse old ones, no newly created
    optimizer, learning_rate, learning_rate_decay_fn = \
          create_optimizer(optimizer, FLAGS.optimizer, FLAGS.learning_rate, num_steps_per_epoch + 1, global_step, num_epochs, finetune_start_step)
    optimizer_results = [[optimizer, learning_rate, learning_rate_decay_fn, None]]
    opts = [optimizer]
    learning_rates_ = [learning_rate]
  
  if FLAGS.round == 0:
    gezi.set_global('opts', opts)
  else:
    # we do not recreate optimizers just reuse but we need to re assign learning rate as it changed like traingle learning rate warmup and step change
    opts = gezi.get_global('opts')
    for i in range(len(opts)):
      lr_set = False
      if hasattr(opts[i], '_lr'):
        opts[i]._lr = learning_rates_[i]
        lr_set = True
      if hasattr(opts[i], '_learning_rate'):
        opts[i]._learning_rate = learning_rates_[i]
        lr_set = True
      assert lr_set

  # TODO 感觉下面如果 train_loop 必须重新构图 因为 learning_rate 变化了 optimizer.lr变化了 不重新构图的话还是走老的图 老的lr变化
  summaries = []
  if FLAGS.monitor_global_gradients:
    summaries += ['global_gradient_norm']
  if FLAGS.monitor_gradients:
    summaries += ['gradients', 'gradient_norm']
  
  if not isinstance(ops[0], (list,tuple)):  
    logging.debug('----------optimzier path 1')
    #https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow
    train_ops = []
    for optimizer, learning_rate, learning_rate_decay_fn, variables in optimizer_results:
      train_op = melt.layers.optimize_loss(
        losses=[ops[0]],
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clip_gradients=FLAGS.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn,
        variables=variables,
        summaries=summaries,
        use_horovod=FLAGS.use_horovod,
        horovod_fp16=FLAGS.horovod_fp16,
        name=optimize_scope)
      train_ops.append(train_op)
      train_op = tf.group(*train_ops)
  else: 
    #---as in cifa10 example, put all but tower loss on cpu, wiki say, that will be faster,
    #but here I find without setting to cpu will be faster..
    #https://github.com/tensorflow/tensorflow/issues/4881
    #I've noticed same thing on cirrascale GPU machines - putting parameters on gpu:0 and using gpu->gpu transfer was a bit faster. I suppose this depends on particular details of hardware -- if you don't have p2p connectivity between your video cards then keeping parameters on CPU:0 gives faster training.
    #err but for my pc no p2p, with PHB connection nvidia-smi topo -m, still hurt by set cpu.. may be should not put cpu here
    update_ops = ops[0][1]
    ops[0] = ops[0][0]
    
    if FLAGS.variable_strategy == 'cpu' and FLAGS.num_gpus and FLAGS.num_gpus > 1:
      logging.debug('----------optimzier path 2, device cpu')
      train_ops = []
      for i, (optimizer, learning_rate, learning_rate_decay_fn, variables) in enumerate(optimizer_results):
        with tf.device('/cpu:0'):
          train_op = melt.layers.optimize_loss(
              losses=ops[0],
              num_gpus=num_gpus,
              global_step=global_step,
              learning_rate=learning_rate,
              optimizer=optimizer,
              clip_gradients=FLAGS.clip_gradients,
              learning_rate_decay_fn=learning_rate_decay_fn,
              variables=variables,
              update_ops=update_ops,
              name=optimize_scope,
              increment_global_step=True if i == 0 else False,
              summaries=summaries,
              use_horovod=FLAGS.use_horovod,
              horovod_fp16=FLAGS.horovod_fp16,
              use_tpu=FLAGS.use_tpu)
          train_ops.append(train_op)
          train_op = tf.group(*train_ops)
    else:
      # mainly here
      logging.debug('----------optimzier path 3')
      train_ops = []
      for i, (optimizer, learning_rate, learning_rate_decay_fn, variables) in enumerate(optimizer_results):
        #print('--------', learning_rate, id(learning_rate))
        train_op = melt.layers.optimize_loss(
            losses=ops[0],
            num_gpus=num_gpus,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=optimizer,
            clip_gradients=FLAGS.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn,
            variables=variables,
            update_ops=update_ops,
            name=optimize_scope,
            increment_global_step=True if i == 0 else False,
            summaries=summaries,
            use_horovod=FLAGS.use_horovod,
            horovod_fp16=FLAGS.horovod_fp16,
            use_tpu=FLAGS.use_tpu)
        train_ops.append(train_op)
        train_op = tf.group(*train_ops)

    # set the last tower loss as loss in ops
    # TODO FIXME how to check if ops[0] here should be scalar ?
    ops[0] = ops[0][-1]


  # TODO now only trac the first learning rate  
  learning_rate = learning_rates_
 
  ops.insert(0, train_op)
  ops.insert(1, learning_rate)
  
  for i, learning_rate in enumerate(learning_rates_):
    #logging.info('-----------{} learning_rate:{}'.format(i, learning_rate))
    lr_name = 'learning_rate' if not i else 'learning_rate_%d' % i
    tf.compat.v1.add_to_collection(lr_name, learning_rate)

    try:
      sess.run(tf.compat.v1.variables_initializer([learning_rate]))
      logging.info('%s inited using sess run' % lr_name, learning_rate)
    except Exception:
      # print(traceback.format_exc(), file=sys.stderr)
      pass

  #-----------post deal
  save_interval_seconds = FLAGS.save_interval_seconds if FLAGS.save_interval_seconds > 0 \
     else FLAGS.save_interval_hours * 3600 

  interval_steps=FLAGS.interval_steps
  valid_interval_steps=FLAGS.valid_interval_steps
  metric_eval_interval_steps=FLAGS.metric_eval_interval_steps
  save_model=FLAGS.save_model 
  save_interval_steps = FLAGS.save_interval_steps 
  num_steps = num_steps or FLAGS.num_steps
  num_epochs = num_epochs or FLAGS.num_epochs

  if not save_interval_steps:
    save_interval_steps = 1e20

  if FLAGS.work_mode == 'train_only' or FLAGS.train_only:
    eval_ops = None 
    metric_eval_fn = None
    logging.debug('running train only mode')
  elif FLAGS.work_mode == 'train_metric':
    eval_ops = None 
    assert metric_eval_fn is not None, 'set metric_eval to 1'
    logging.debug('running train+metric mode')
  elif FLAGS.work_mode == 'train+valid':
    metric_eval_fn = None
    logging.debug('running train+valid mode')
  elif FLAGS.work_mode.startswith('test'):
    ops = None
    logging.debug('running test only mode')
    interval_steps = 0
    valid_interval_steps = 1
    metric_eval_interval_steps /= FLAGS.valid_interval_steps
    save_model = False
  elif FLAGS.work_mode.startswith('metric') or FLAGS.work_mode.startswith('eval') or FLAGS.work_mode.startswith('valid') or gezi.env_has('METRIC'):
    #TODO name is a bit cofusing for mode, eval or metric means using metric evaluation
    #test above means using eval_loss(valid_loss) as composed to train_loss for evaluation
    ops = None 
    eval_ops = None
    if FLAGS.work_mode == 'valid+test':
      logging.debug('runing valid and test only mode')
    else:
      logging.debug('running metric eval only mode')
    interval_steps = 0 
    valid_interval_steps = 1
    metric_eval_interval_steps /= FLAGS.valid_interval_steps    
    save_model = False
    assert metric_eval_fn is not None 

  if FLAGS.work_mode.endswith('once'):
    num_epochs = -1 #hack to make only do one step!

  #TODO hack seq2seq/OptimizeLoss/seq2seq/main/decode/rnn/basic_lstm_cell/kernel/Adagrad (DT_FLOAT) [1280,4096] need to save
  if variables_to_save is not None:
    optimize_vars = set(slim.get_variables(get_global_scope() + '/OptimizeLoss'))
    assert optimize_vars, 'optimizer must has scope %s'%(get_global_scope() + '/OptimizeLoss')
    variables_to_save = list(set(variables_to_save) | optimize_vars)

  if output_collection_names is None and FLAGS.freeze_graph_collections:
    if model is not None and hasattr(model, 'init_predict'):
      if not hasattr(model, 'inited_predict') or not model.inited_predict:
        model.init_predict()
        model.inited_predict = True

    all_keys = sess.graph.get_all_collection_keys()
    exclude_keys = set(['variables', 'queue_runners', 'summaries', 'train_op', 'update_ops', 'model_variables', 'cond_context', 'while_context'])
    output_collection_names = [x for x in all_keys if x not in exclude_keys and not 'train' in x and not x.endswith('_end_points')]
  logging.debug('all collection keys: {}'.format(all_keys[:100]))
  logging.debug('collection names to freeze: {}'.format(output_collection_names))

  logging.debug('ops', ops)
  logging.debug('eval_ops', eval_ops)

  if FLAGS.learning_rate_patience:
    assert metric_eval_fn is not None, 'need to use metrci eval fn to monitor and decay learning rate'

  restore_include = restore_include or FLAGS.restore_include.split(',') if FLAGS.restore_include else None
  restore_exclude = restore_exclude or FLAGS.restore_exclude.split(',') if FLAGS.restore_exclude else None

  return melt.flow.train_flow(
             ops, 
             names=names,
             gen_feed_dict_fn=gen_feed_dict_fn,
             deal_results_fn=deal_results_fn,
             # TODO horovod might multiple gpu eval then reduceall mean
             #eval_ops=eval_ops if not FLAGS.use_horovod or hvd.rank() == 0 else None,
             eval_ops=eval_ops,
             eval_names=eval_names,
             gen_eval_feed_dict_fn=gen_eval_feed_dict_fn,
             deal_eval_results_fn=deal_eval_results_fn,
             interval_steps=interval_steps,
             valid_interval_steps=valid_interval_steps,
             eval_loops=FLAGS.eval_loops,
             num_epochs=num_epochs,
             num_steps=num_steps,
             save_interval_seconds=save_interval_seconds,
             save_interval_steps=save_interval_steps,
             save_model=save_model if not FLAGS.use_horovod or hvd.rank() == 0 else False,
             save_interval_epochs=FLAGS.save_interval_epochs,
             freeze_graph=FLAGS.freeze_graph,
             #optimizer=optimizer, 
             optimizer=None, #must set None since here we have done choosing optimizer
             learning_rate=learning_rate,
             learning_rate_patience=FLAGS.learning_rate_patience,
             learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
             num_steps_per_epoch=num_steps_per_epoch,
             max_models_keep=FLAGS.max_models_keep,
             #model_dir=model_dir if not FLAGS.use_horovod or hvd.rank() == 0 else None,
             # still pass model dir eve hvd rank > 0 for evaluate or infer might use it 
             model_dir=model_dir,
             log_dir=log_dir if not FLAGS.use_horovod or hvd.rank() == 0 else None,
             restore_from_latest=FLAGS.restore_from_latest,
             #metric_eval_fn=metric_eval_fn if not (FLAGS.use_horovod and not FLAGS.horovod_eval) or hvd.rank() == 0 else None,  
             metric_eval_fn=metric_eval_fn,
             metric_eval_interval_steps=metric_eval_interval_steps,
             valid_interval_epochs=FLAGS.valid_interval_epochs,
             first_interval_epoch=FLAGS.first_interval_epoch,
             #inference_fn=inference_fn if not (FLAGS.use_horovod and not FLAGS.horovod_eval) or hvd.rank() == 0 else None,
             inference_fn=inference_fn,
             inference_interval_epochs=FLAGS.inference_interval_epochs,
             no_log=FLAGS.no_log,
             summary_excls=summary_excls,
             init_fn=init_fn,
             restore_fn=restore_fn,
             restore_include=restore_include,
             restore_exclude=restore_exclude,
             save_all_scope=save_all_scope,
             variables_to_restore=variables_to_restore,
             variables_to_save=variables_to_save,
             output_collection_names=output_collection_names,
             output_node_names=output_node_names,
             write_during_train=FLAGS.write_during_train,
             use_horovod=FLAGS.use_horovod,
             model=model,
             callbacks=callbacks,
             sess=sess)

# TODO evaluate can only infer if eval_fn is None
def evaluate(ops, iterator, num_steps, num_examples, eval_fn, 
             model_path=None, names=None, write_fn=None, write_streaming=False,
             num_gpus=1, write=False, write_valid_only=False, ofile=None,
             suffix='.valid.csv', sep=',', keys=None, sess=None): 
  
  if not FLAGS.do_valid:
    return [], []

  comm = gezi.get_global('dist').comm
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    import horovod.tensorflow as hvd

  rank = FLAGS.local_rank

  ## TODO 检查一下 什么时候需要 分开每个worker写一个文件？ 当前比较常用horovod 会分开计算 最后汇总 只是worker 0 写结果
  # if FLAGS.world_size > 1:
  #   suffix = suffix.replace('.csv', f'-{FLAGS.local_rank}.csv')
  #   if ofile:
  #     ofile = ofile.replace('.csv', f'-{FLAGS.local_rank}.csv')

  if not write_fn:
    write_streaming = True

  if not keys:
    keys = ['id']

  if isinstance(keys, str):
    keys = keys.split(',')  

  ids_list = [[] for _ in range(len(keys))]
  predictions_list = []
  labels_list = []
  other_ = ops[0][-1]
  others_list = [[] for _ in range(len(other_))]

  if write_fn:
    kwargs_write = {} 
    write_args = inspect.getfullargspec(write_fn).args 

  if not sess:
    sess = melt.get_session()
  try:
    sess.run(iterator.initializer)
  except Exception:
    pass

  try:
    for i in range(num_gpus):
      if isinstance(ops[i][0], dict):
        ops[i][0] = [ops[i][0][key] for key in keys]
  except Exception:
    traceback.print_exc()
    pass

  if FLAGS.profile_interval_steps:
    # https://blog.csdn.net/kenneth_yu/article/details/77466776
    profiler = model_analyzer.Profiler(graph=sess.graph)
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    timeline_dir = f'{FLAGS.model_dir}/timeline_valid'
    os.system(f'mkdir -p {timeline_dir}')

  import psutil
  
  timer_ = gezi.Timer()
  try:
    # here for multiple gpu (not horovod) dataset is repeate mode
    desc = 'eval' if not FLAGS.valid_hour else f'{FLAGS.valid_hour}-{FLAGS.eval_round}-eval'
    for step in tqdm(range(num_steps), total=num_steps, desc=desc, ascii=False):
      feed_dict={}
      kwargs = {}
      if FLAGS.profile_interval_steps and (step + 1) % FLAGS.profile_interval_steps == 0:
        kwargs['options'] = run_options 
        kwargs['run_metadata'] = run_metadata
      
      results = sess.run(ops, feed_dict=feed_dict, **kwargs)
      
      if FLAGS.profile_interval_steps and (step + 1) % FLAGS.profile_interval_steps == 0:
        profile_step(profiler, step + 1, run_metadata, timeline_dir)
      
      for i in range(num_gpus):
        ids, labels, predictions, others = results[i]
        # ids[0] = gezi.decode(ids[0])
        for j in range(len(keys)):
          ids_list[j].append(gezi.squeeze(ids[j])) 
        predictions_list.append(gezi.squeeze(predictions))
        labels_list.append(gezi.squeeze(labels))
        for i, key in enumerate(others): 
          others_list[i].append(gezi.squeeze(others[key]))
  except tf.errors.OutOfRangeError:
    pass
  FLAGS.valid_time = timer_.elapsed_minutes(reset=False)

  if FLAGS.mode == 'valid_perf':
    valid_perf_file = f'{FLAGS.model_dir}/valid_perf.csv'
    try:
      df = pd.read_csv(valid_perf_file)
    except Exception:
      df = pd.DataFrame()
    seconds = timer_.elapsed(reset=False)
    minute = int(int(seconds) / 60)
    second = int(seconds) % 60
    res = {'model':FLAGS.model_name, 'path': os.path.basename(FLAGS.model_dir),
           'duration': f'{minute}:{second}', 'steps/s': num_steps / seconds, 
           'insts/s': num_examples / seconds}
    df = df.append(res, ignore_index=True)
    df = df[['model', 'path', 'duration', 'steps/s', 'insts/s']]
    gezi.pprint_df(df, print_fn=logging.info)
    df.to_csv(valid_perf_file, index=False, float_format='%.2f')
    exit(0)

  # TODO below a bit slow for merging results
  if FLAGS.use_horovod and FLAGS.horovod_eval:
    sess.run(hvd.allreduce(tf.constant(0)))
    # here for horovod mutliple gpu dataset is not repeat mode 
    ids_list = Manager().list(ids_list)
    for i in range(len(keys)):
      ids_list[i] = comm.allgather(np.concatenate(ids_list[i]))
    predictions_list = comm.allgather(np.concatenate(predictions_list))
    labels_list = comm.allgather(np.concatenate(labels_list))
    for i in range(len(others_list)):
      others_list[i] = comm.allgather(np.concatenate(others_list[i]))
    comm.barrier()

    ids = [np.concatenate(ids_list[i]) for i in range(len(keys))]
    predicts = np.concatenate(predictions_list)
    labels = np.concatenate(labels_list)
    others = [np.concatenate(others_list[i]) for i in range(len(others_list))]

    # below is for batch parse which if not repeat mode then final batch will still same size not smaller
    # and not use repeat mode so last batch fill with id '' empty we can remove here
    if FLAGS.batch_parse:     
      filter_ids = ids[0] != ''
      predicts = predicts[filter_ids]
      labels = labels[filter_ids]
      for i in range(len(keys)):
        ids[i] = ids[i][filter_ids]
      for i in range(len(others)):
        others[i] = others[i][filter_ids]
     
    if not FLAGS.num_valid:  # if set num_valid might be debug purpose for smaller data quick run
      assert len(predicts) > 0, 'all ids are empty string ? we ignore these instance with empty id'
      assert len(predicts) == num_examples, 'num predicts:%d  num_examples:%d, maybe in batch_parse mode and not set FLAGS.batch_parse==True' % (len(predicts), num_examples)
  else:
    ids = [np.concatenate(ids_list[i])[:num_examples] for i in range(len(keys))]
    predicts = np.concatenate(predictions_list)[:num_examples]
    labels = np.concatenate(labels_list)[:num_examples]
    others = [np.concatenate(others_list[i])[:num_examples] for i in range(len(others_list))]
  
  if FLAGS.work_mode != 'train':
    sess.close()
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

  # with gezi.Timer('----afsg', print_fn=print) as timer:
  other = {}
  # for i, key in enumerate(other_): 
  #   other[key] = gezi.decode(others[i])
  for i, key in enumerate(other_): 
    other[key] = others[i]

  # for i in range(len(ids)):
  #   ids[i] = gezi.decode(ids[i])
  
  kwargs = {}
  args = inspect.getfullargspec(eval_fn).args    
  if 'model_path' in args:
    kwargs['model_path'] = model_path
  if 'ids' in args:
    kwargs['ids'] = ids[0]
  if 'info' in args:
    kwargs['info'] = dict(zip(keys, ids))
  if 'x' in args:
    kwargs['x'] = dict(zip(keys, ids))
  if 'other' in args:
    kwargs['other'] = other
  if 'others' in args:
    kwargs['others'] = other

  dic = Manager().dict()

  def _eval():
    # TODO check here, even with comm.barrier might still hang
    if not write_valid_only:
      try:
        results = eval_fn(labels, predicts, **kwargs)
      except Exception:
        logging.warning('eval fn error')
        logging.warning(traceback.format_exc())
        results = [], []
      dic['names'], dic['vals'] = results if isinstance(results, (list, tuple)) else list(zip(*results.items()))

  if model_path and not ofile:
    ofile = model_path + suffix
    
  def _write():
    if rank != 0:
      return
    if ofile and write:
      logging.info(f'write {len(predicts)} valid result for each valid instance to {ofile}')
      if write_streaming:
        with open(ofile, 'w') as out:
          if names:
            print(*names, sep=sep, file=out)
          for i, (id, label, predict) in tqdm(enumerate(zip(zip(*ids), labels, predicts)), total=len(labels), ascii=False):
            id = sep.join(map(str, id))
            if write_fn is None:
              if not gezi.iterable(label):
                label = [label]
              if not gezi.iterable(predict):
                predict = [predict] 
              print(id, *label, *predict, sep=sep, file=out)
            else:
              if 'others' in write_args:
                kwargs_write['others'] = dict([(key, other[key][i]) for key in other])
              write_fn(id, label, predict, out, **kwargs_write)
      else: 
        if 'ids' in write_args:
          ids = ids[0] if len(keys) == 1 else kwargs['x']
          if 'others' in write_args:
            kwargs_write['others'] = dict([(key, other[key]) for key in other])
          write_fn(ids, labels, predicts, ofile, **kwargs_write)
        else:
          for i, (id, label, predict) in tqdm(enumerate(zip(zip(*ids), labels, predicts)), total=len(labels), ascii=False):
            id = sep.join(map(str, id))
            if 'others' in write_args:
              kwargs_write['others'] = dict([(key, other[key][i]) for key in other])
            write_fn(id, label, predict, ofile, **kwargs_write)

  # if rank == 0:
  if FLAGS.use_pymp:
    funcs = [_eval, _write]
    with pymp.Parallel(len(funcs)) as p:
      for i in p.range(len(funcs)):
        funcs[i]()
  else:
    _eval()
    _write()

  if 'names' in dic and 'vals' in dic:
    names, vals = dic['names'], dic['vals']
  else:
    names, vals = [], []
  
  if comm:
    comm.barrier()
  
  return names, vals

# TODO inference modify according to evaluate
def inference(ops, iterator, num_steps, num_examples, 
              model_path=None, names=None, debug_names=None,
              write_fn=None, write_streaming=False, num_gpus=1, 
              ofile=None, suffix='.infer', sep=',', sess=None):

  if not FLAGS.do_test:
    return 
    
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    comm = gezi.get_global('dist').comm
    import horovod.tensorflow as hvd
  
  if not write_fn:
    write_streaming = True
  ids_list = []  
  predictions_list = []

  other_ = ops[0][-1]
  others_list = [[] for _ in range(len(other_))]

  if not sess:
    sess = melt.get_session()
  
  try:
    sess.run(iterator.initializer)
  except Exception:
    pass

  try:
    for i in range(num_gpus):
      if isinstance(ops[i][0], dict):
        ops[i][0] = ops[i][0]['id']
  except Exception:
    pass

  try:
    for _ in tqdm(range(num_steps), total=num_steps, desc='infer', ascii=False):
      feed_dict = {}
      # feed_dict[K.learning_phase()] = 0
      melt.test_feed_dict.update(feed_dict)
      results = sess.run(ops, feed_dict=melt.test_feed_dict)
      for i in range(num_gpus):
        ids, predictions, others = results[i]
        ids = gezi.squeeze(ids)
        predictions = gezi.squeeze(predictions)
        # ids = gezi.decode(ids)     
        ids_list.append(ids)   
        predictions_list.append(predictions)
        for i, key in enumerate(others): 
          others_list[i].append(gezi.squeeze(others[key]))
  except tf.errors.OutOfRangeError:
    logging.warning(traceback.format_exc())


  if FLAGS.use_horovod and FLAGS.horovod_eval:
    sess.run(hvd.allreduce(tf.constant(0)))
    # here for horovod mutliple gpu dataset is not repeat mode 
    ids_list = comm.allgather(np.concatenate(ids_list))
    predictions_list = comm.allgather(np.concatenate(predictions_list))
    for i in range(len(others_list)):
      others_list[i] = comm.allgather(np.concatenate(others_list[i]))
    comm.barrier()

    ids = np.concatenate(ids_list)[:num_examples]
    predicts = np.concatenate(predictions_list)[:num_examples]
    # labels = np.concatenate(labels_list)
    others = [np.concatenate(others_list[i])[:num_examples] for i in range(len(others_list))]

    # # below is for batch parse which if not repeat mode then final batch will still same size not smaller
    # # and not use repeat mode so last batch fill with id '' empty we can remove here
    # if FLAGS.batch_parse:     
    #   filter_ids = ids[0] != ''
    #   predicts = predicts[filter_ids]
    #   labels = labels[filter_ids]
    #   for i in range(len(keys)):
    #     ids[i] = ids[i][filter_ids]
    #   for i in range(len(others)):
    #     others[i] = others[i][filter_ids]
     
    # if not FLAGS.num_valid:  # if set num_valid might be debug purpose for smaller data quick run
    #   assert len(predicts) > 0, 'all ids are empty string ? we ignore these instance with empty id'
    #   assert len(predicts) == num_examples, 'num predicts:%d  num_examples:%d, maybe in batch_parse mode and not set FLAGS.batch_parse==True' % (len(predicts), num_examples)
  else:
    ids = np.concatenate(ids_list)[:num_examples]
    # ids = [np.concatenate(ids_list[i])[:num_examples] for i in range(len(keys))]
    #print(predictions_list)
    predicts = np.concatenate(predictions_list)[:num_examples]
    # labels = np.concatenate(labels_list)[:num_examples]
    others = [np.concatenate(others_list[i])[:num_examples] for i in range(len(others_list))]

  # # TODO for infer might not need to use all gather ...
  # if FLAGS.use_horovod and FLAGS.horovod_eval:
  #   sess.run(hvd.allreduce(tf.constant(0)))
  #   ids_list = comm.allgather(np.concatenate(ids_list))
  #   predictions_list = comm.allgather(np.concatenate(predictions_list))
  #   for i in range(len(others_list)):
  #     others_list[i] = comm.allgather(np.concatenate(others_list[i]))
  #   comm.barrier()
  #   ids = np.concatenate(ids_list)
  #   predicts = np.concatenate(predictions_list)
  #   others = [np.concatenate(others_list[i]) for i in range(len(others_list))]

  #   if FLAGS.batch_parse:     
  #     filter_ids = ids != ''
  #     predicts = predicts[filter_ids]
  #     ids = ids[filter_ids]
    
  #   assert len(predicts) > 0, 'all ids are empty string ? we ignore these instance with empty id'
  #   assert len(predicts) == num_examples, 'num predicts:%d  num_examples:%d' % (len(predicts), num_examples)
  # else:
  #   ids = np.concatenate(ids_list)[:num_examples]
  #   predicts = np.concatenate(predictions_list)[:num_examples]
  #   others = [np.concatenate(others_list[i])[:num_examples] for i in range(len(others_list))]

  if FLAGS.work_mode != 'train':
    sess.close()
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

  if (not FLAGS.use_horovod or hvd.rank() == 0):
    if not ofile:
      ofile = FLAGS.model_dir + '/submission.csv'

    other = {}
    for i, key in enumerate(other_): 
      other[key] = others[i]
    
    if write_streaming:
      with open(ofile, 'w') as out:
        if names:
          print(*names, sep=sep, file=out)
        for i, (id, predict) in tqdm(enumerate(zip(ids, predicts)), total=len(ids), ascii=False):
          if write_fn is None:
            if not gezi.iterable(predict):
              predict = [predict]
            print(id, *predict, sep=sep, file=out)
          else:
            kwargs_write = {}
            write_args = inspect.getfullargspec(write_fn).args 
            if 'others' in write_args:
              kwargs_write['others'] = dict([(key, other[key][i]) for key in other])
            write_fn(id, predict, out, **kwargs_write)
    else:
      kwargs_write = {}
      write_args = inspect.getfullargspec(write_fn).args 
      if 'others' in write_args:
        kwargs_write['others'] = other
      write_fn(ids, predicts, ofile, **kwargs_write)

def train(model, 
          loss_fn, 
          Dataset=None,
          dataset=None,  # train dataset
          valid_dataset=None,  # Actually this is eval dataset for full evaluation on full valid data
          valid_dataset2=None, # This is valid dataset actuall compare to train dataset, which is validation on single batch each run 
          test_dataset=None,
          eval_info_dataset=None,
          test_info_dataset=None,
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          eval_keys=[],
          init_fn=None,
          restore_fn=None,
          write_valid=None,
          valid_names=None,
          infer_names=None,
          infer_debug_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid.csv',
          infer_suffix='.infer.csv',
          write_streaming=False,
          optimizer=None,
          variables_list_fn=None,
          weights=1.0, # instance weights for loss 最好loss内部自己处理weight
          sep='\t',
          out_hook=None,
          out_keys=[],
          callbacks=[],
          metrics=[],
          initial_epoch=0,
          return_info=False,
          dry_run=False):
  
  write_valid = write_valid if write_valid is not None else FLAGS.write_valid
  rank = FLAGS.local_rank

  metric_eval_fn, inference_fn = None, None

  if FLAGS.round == 0 or (not FLAGS.feed_dataset):
    timer = gezi.Timer('count_examples, build_graph, preparing for train_flow')
    if weights is None:
      weights = 1.0

    if Dataset is None:
      assert dataset

    if not hasattr(model, 'eval_keys'):
      model.eval_keys = eval_keys
    else:
      model.eval_keys = model.eval_keys or eval_keys
    eval_keys = model.eval_keys

    if not hasattr(model, 'out_keys'):
      model.out_keys = out_keys
    else:
      model.out_keys = model.out_keys or out_keys
    out_keys = model.out_keys   
    logging.debug('eval_keys:', model.eval_keys, 'out_keys:', model.out_keys)
    
    input_ =  FLAGS.train_input 
    try:
      inputs = Dataset.get_filenames_('train')
    except Exception:
      inputs = gezi.list_files(input_)
    # assert inputs, input_
    # inputs.sort()

    all_inputs = inputs

    batch_size = melt.batch_size() if not FLAGS.use_horovod and not FLAGS.ps_strategy else FLAGS.batch_size
    num_gpus = melt.num_gpus()
    valid_batch_size = melt.eval_batch_size() if not FLAGS.use_horovod and not FLAGS.ps_strategy else FLAGS.eval_batch_size
    test_batch_size = valid_batch_size

    # hack for horovod
    if FLAGS.use_horovod: 
      num_gpus = 1
      import horovod.tensorflow as hvd

    logging.debug('model', model, 'Dataset', Dataset, 'loss_fn', loss_fn)

    if num_gpus > 1:
      assert not FLAGS.batch_sizes, 'Not support batch sizes for num gpus > 1, TODO'

    valid_inputs = None
    num_valid_examples = None
    if FLAGS.fold is not None:
      def _check(inputs):
        mark = 'no'
        has_train, has_valid, has_other = False, False, False
        # all train_mark, all valid_mark, train and valid mark, no mark
        if FLAGS.train_mark:
          for x in inputs:
            if FLAGS.train_mark in x:
              has_train = True
            else:
              if not FLAGS.valid_mark or FLAGS.valid_mark not in x:
                has_other = True
        if FLAGS.valid_mark:
          for x in inputs:
            if FLAGS.valid_mark in x:
              has_valid = True
            else:
              if not FLAGS.train_mark or FLAGS.train_mark not in x:
                has_other = True
    
        if has_train and has_valid:
          mark = 'train_valid'
        elif has_other and has_valid:
          mark = 'other_valid'
        elif has_train:
          mark = 'train'
        elif has_valid:
          mark = 'valid'
        return mark
      mark = _check(inputs)
      def _is_train(file_path, mark):
        if mark == 'train_valid' or mark == 'other_valid':
          if FLAGS.valid_mark not in file_path:
            return True
        return not file_path.rstrip('/').split('/')[-1].startswith('record_%d.' % FLAGS.fold)
      inputs = [x for x in inputs if _is_train(x, mark)]
      if not FLAGS.test_aug:
        valid_inputs = [x for x in all_inputs if 'aug' not in x and FLAGS.valid_exclude not in x and x not in inputs]
      else:
        valid_inputs = [x for x in all_inputs if 'aug' in x and FLAGS.valid_exclude not in x and x not in inputs]
      logging.debug('valid_inputs:', len(valid_inputs), valid_inputs[:10])
    logging.debug('inputs:', len(inputs), inputs[:10])

    num_folds = FLAGS.num_folds or len(inputs) + 1

    iter = None
    num_steps_per_epoch = None
    if dataset is None:
      dataset = Dataset('train')
      iter = dataset.make_batch(batch_size, inputs, repeat=True, initializable=True, 
                                hvd_shard=FLAGS.horovod_shard, simple_parse=FLAGS.simple_parse,
                                cache=FLAGS.cache or FLAGS.cache_train, cache_file=FLAGS.cache_file,
                                world_size=FLAGS.world_size, rank=FLAGS.local_rank)
      gezi.set_global('iter', iter)
    else: 
      iter = dataset

    num_test_examples = None
    num_test_steps_per_epoch = None
    try:
      num_examples = len(dataset)
    except Exception:
      num_examples = 1000000
    if FLAGS.fold is not None:
      num_valid_examples = melt.get_num_records(valid_inputs)
      num_examples = num_examples - num_valid_examples
    if FLAGS.num_train:
      num_examples = min(num_examples, FLAGS.num_train)
    if num_examples:
      num_steps_per_epoch = -(-num_examples // melt.batch_size())
    else:
      num_steps_per_epoch = None
    logging.info('num_train_examples:', num_examples)

    if FLAGS.train_loop and not num_examples or num_examples < 1:
      return None

    if FLAGS.valid_input:
      valid_inputs = gezi.list_files(FLAGS.valid_input)

    test_iter = None
    valid_iter = None
    valid_iter2 = None
    num_valid_steps_per_epoch = None
    if valid_inputs:
      if valid_dataset is None: 
        valid_dataset = Dataset('valid')
        # here not set repeat to False, for if multiple gpu might repeat is True we can stop using num steps
        repeat = False if not FLAGS.drop_remainder else True
        valid_iter = valid_dataset.make_batch(valid_batch_size, valid_inputs, subset='valid', repeat=repeat, initializable=True, hvd_shard=FLAGS.horovod_eval,
                                              cache=FLAGS.cache or FLAGS.cache_valid, cache_file=FLAGS.cache_file,
                                              world_size=FLAGS.world_size, rank=FLAGS.local_rank)
        gezi.set_global('valid_iter', valid_iter)
        # valid iter2 for valid op
        if FLAGS.use_valid_loss:
          valid_iter2 = valid_dataset.make_batch(valid_batch_size, valid_inputs, subset='valid', repeat=True, initializable=True, hvd_shard=FLAGS.horovod_eval,
                                                 cache=FLAGS.cache or FLAGS.cache_valid, cache_file=FLAGS.cache_file,
                                                 world_size=FLAGS.world_size, rank=FLAGS.local_rank)
          gezi.set_global('valid_iter2', valid_iter2)
        # print(valid_inputs, FLAGS.use_valid_loss)
        # exit(0)
      else:
        if FLAGS.use_valid_loss:
          valid_iter2 = valid_dataset2
        valid_iter = valid_dataset
      logging.debug('valid_inputs:', len(valid_inputs), valid_inputs[:10])
    else:
      valid_dataset = None
    
    if valid_dataset is not None:
      try:
        num_fll_valid_examples = len(valid_dataset)
        num_valid_examples = num_valid_examples or num_fll_valid_examples
      except Exception:
        num_valid_examples = 1000000
      if FLAGS.num_valid:
        num_valid_examples = min(num_valid_examples, FLAGS.num_valid)
      try:
        gezi.set('num_full_valid_examples', num_fll_valid_examples)
        gezi.set('num_full_valid_steps_per_epoch', -(-num_full_valid_examples // melt.eval_batch_size()))
      except Exception:
        pass
      num_valid_steps_per_epoch = -(-num_valid_examples // melt.eval_batch_size()) if num_valid_examples else None    
      logging.info('num_valid_examples:', num_valid_examples)

    if FLAGS.test_input:
      test_inputs = gezi.list_files(FLAGS.test_input)
      logging.debug('test_inputs:', len(test_inputs), test_inputs[:10])
    else:
      test_inputs = None
   
    num_test_examples = None
    if test_inputs:
      if test_dataset is None:
        test_dataset = Dataset('test')
        test_iter = test_dataset.make_batch(test_batch_size, test_inputs, subset='test', hvd_shard=FLAGS.horovod_eval,
                                            repeat=False, initializable=True,
                                            cache=FLAGS.cache or FLAGS.cache_test, cache_file=FLAGS.cache_file,
                                            world_size=FLAGS.world_size, rank=FLAGS.local_rank)
      else:
        test_iter = test_dataset
      num_test_examples = FLAGS.num_test or len(test_dataset)
      num_test_steps_per_epoch = -(-num_test_examples // melt.eval_batch_size()) if num_test_examples else None
      logging.info('num_test_examples:', num_test_examples)
    else:
      test_dataset = None

    if not FLAGS.train_loop:
      assert not (FLAGS.valid_input and not num_valid_examples), FLAGS.valid_input
    else:
      if (FLAGS.valid_input and not num_valid_examples):
        logging.warning('valid_input', FLAGS.valid_input, num_valid_examples, 'ignore this round')
        return None

    scope_name = ''
    if hasattr(iter, 'get_next'):
      batch = iter.get_next()
    else:
      batch = iter

    logging.debug('melt.split_batch')
    x, y = melt.split_batch(batch, batch_size, num_gpus)

    def _get_kwargs(x):
      kwargs = {}
      if 'x' in inspect.getfullargspec(loss_fn).args:
        kwargs['x'] = x 
      if 'model' in inspect.getfullargspec(loss_fn).args:
        kwargs['model'] = model
      if 'weights' in inspect.getfullargspec(loss_fn).args:
        weights_ = x[weights] if isinstance(weights, str) else weights
        kwargs['weights'] = weights_
      if 'weight' in inspect.getfullargspec(loss_fn).args:
        weights_ = x[weights] if isinstance(weights, str) else weights
        kwargs['weight'] = weights_
      return kwargs

    def train_fn(x, y):
      model.mode = 'train'
      try:
        if 'training' in inspect.getfullargspec(model.call).args:
          y_ = model(x, training=True)
        else:
          if hasattr(model, 'train'):
            model.train()
          y_ = model(x)
      except Exception:
        assert FLAGS.work_mode != 'train' or FLAGS.num_steps < 0
        # TODO only dum infer graph without train_input
        if not hasattr(model, 'inited_predict') or not model.inited_predict:
          model.init_predict()
          model.inited_predict = True
        FLAGS.is_infer = False
        y_ = model(model.input_feed)
        FLAGS.is_infer = True
      kwargs = _get_kwargs(x)
      loss = loss_fn(y, y_, **kwargs)
      if FLAGS.l2_weight_decay:
        loss += melt.losses.l2_loss() * FLAGS.l2_weight_decay
      return loss

    logging.debug('build graph, melt.tower')
    # TODO FIXME ! this will fix dropout problem for imdb dataset but for image classification like kaggle blindness adding this will not convergent might due to batch norm
    # if you in training mode might overfit quickly , train loss decrease ok, might due to not fix trainable for tf model.layer.train_able=False not work  
    # Anyway should add set learning phrase here without it dropout always in valid mode not work easy to overfit
    K.set_learning_phase(1)  
    loss = melt.tower(lambda i: train_fn(x[i], y[i]), num_gpus)
    K.set_learning_phase(0)

    def valid_fn(x, y):
      model.mode = 'valid'
      kwargs = _get_kwargs(x)
      y_ = model(x)
      loss = loss_fn(y, y_, **kwargs)
      return loss

    logging.debug('prepare for metric eval')
    ops = [loss]
    eval_ops = None 
    metric_eval_fn = None
    # if FLAGS.valid_input and valid_dataset is not None:
    if valid_dataset is not None:
      if hasattr(valid_iter2, 'get_next'):
        valid_batch2 = valid_iter2.get_next()
      else:
        valid_batch2 = valid_iter2
      if FLAGS.run_valid_op:
        valid_x2, valid_y2 = melt.split_batch(valid_batch2, valid_batch_size, num_gpus, training=False)
        valid_loss = melt.tower(lambda i: valid_fn(valid_x2[i], valid_y2[i]), num_gpus, training=False)
        valid_loss = tf.reduce_mean(input_tensor=valid_loss)
        eval_ops = [valid_loss]
 
      if hasattr(valid_iter, 'get_next'):
        valid_batch = valid_iter.get_next()
      else: 
        valid_batch = valid_iter
      valid_x, valid_y = melt.split_batch(valid_batch, valid_batch_size, num_gpus, training=False)

      if not valid_names and infer_names:
        valid_names = [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:]

      logging.debug('eval_fn', eval_fn)
      
      def valid_fn(i):
        model.mode = 'valid'
        if hasattr(model, 'eval'):
          model.eval()
        valid_predict = model(valid_x[i])
        others = out_hook(model) if out_hook else melt.out_hook(model, out_keys)
        logging.sinfo(others)
        return valid_x[i], valid_y[i], valid_predict, others

      valid_ops = melt.tower(valid_fn, num_gpus, training=False)
      if eval_fn:
        if not FLAGS.valid_hour:
          ofile = None
        else:
          odir = f'{FLAGS.log_dir}/infos/{FLAGS.valid_hour}' if not FLAGS.loop_fixed_valid else f'{FLAGS.log_dir}/infos/{FLAGS.train_hour}'
          os.system(f'mkdir -p {odir}')
          ofile = f'{odir}/valid.csv'
        metric_eval_fn = lambda model_path=None, write=write_valid: \
                                      evaluate(valid_ops, 
                                              valid_iter,
                                              num_steps=num_valid_steps_per_epoch,
                                              num_examples=num_valid_examples,
                                              eval_fn=eval_fn,
                                              names=valid_names,
                                              write_fn=valid_write_fn,
                                              model_path=model_path,
                                              write=write,
                                              num_gpus=num_gpus,
                                              suffix=valid_suffix,
                                              ofile=ofile,
                                              write_streaming=write_streaming,
                                              write_valid_only=FLAGS.write_valid_only,
                                              keys=eval_keys,
                                              sep=sep)
        # if rank != 0:
          # metric_eval_fn = None

    if FLAGS.test_input and test_dataset is not None:
      if hasattr(test_iter, 'get_next'):
        test_batch = test_iter.get_next()
      else:
        test_batch = test_iter
      test_x, test_y = melt.split_batch(test_batch, test_batch_size, num_gpus, training=False)

      def infer_fn(i):
        model.mode = 'test'
        if hasattr(model, 'eval'):
          model.eval()
        test_predict = model(test_x[i])
        others = out_hook(model) if out_hook else {}
        return test_x[i], test_predict, others

      test_ops = melt.tower(infer_fn, num_gpus, training=False)

      inference_fn = lambda model_path=None: \
                                    inference(test_ops, 
                                              test_iter,
                                              num_steps=num_test_steps_per_epoch,
                                              num_examples=num_test_examples,
                                              names=infer_names,
                                              debug_names=infer_debug_names,
                                              write_fn=infer_write_fn,
                                              model_path=model_path,
                                              num_gpus=num_gpus,
                                              suffix=infer_suffix,
                                              write_streaming=write_streaming,
                                              sep=sep)
      # TODO whey here ?
      # if rank != 0:
      #   inference_fn = None
    else:
      inference_fn = None

    assert num_examples
    
    # from husky.callbacks import EvalCallback
    # val_callback = EvalCallback(valid_iter, eval_fn, num_steps_per_epoch, num_valid_steps_per_epoch, 
    #                              eval_keys, out_hook, write_valid=write_valid)
    # val_callback.set_model(model)
    # init_op = tf.group(tf.global_variables_initializer(), #variables_initializer(global_variables())
    #                  tf.local_variables_initializer()) #variables_initializer(local_variables())
    # melt.get_session().run(tf.global_variables_initializer())
    # val_callback.eval()
    if return_info:
      info = {
        'dataset': iter, 
        'eval_dataset': valid_iter,
        'valid_dataset': valid_iter2,
        'test_dataset': test_iter,
        'num_examples': num_examples,
        'num_valid_examples': num_valid_examples,
        'num_test_examples': num_test_examples,
        'num_steps_per_epoch': num_steps_per_epoch,
        'num_valid_steps_per_epoch': num_valid_steps_per_epoch,
        'num_test_steps_per_epoch': num_test_steps_per_epoch
      }
      return info 

    if ops:
      gezi.set_global('ops', ops.copy())
    if eval_ops:
      gezi.set_global('eval_ops', eval_ops.copy())
    if valid_dataset is not None:
      gezi.set_global('valid_ops', valid_ops)
      gezi.set_global('valid_iter', valid_iter)
    if test_dataset is not None:
      gezi.set_global('test_ops', test_ops)
      gezi.set_global('test_iter', test_iter)
    gezi.set_global('num_gpus', num_gpus)      
    timer.print()
  else:
    ops = gezi.get_global('ops', None)
    if ops:
      ops = ops.copy()
    eval_ops = gezi.get_global('eval_ops', None)
    if eval_ops:
      eval_ops = eval_ops.copy()
    num_gpus = gezi.get_global('num_gpus')
    num_examples = Dataset.num_examples_per_epoch('train')
    num_examples = FLAGS.num_train or num_examples
    num_steps_per_epoch = -(-num_examples // melt.batch_size()) if num_examples else None

    if FLAGS.valid_input:
      num_valid_examples = FLAGS.num_valid or Dataset.num_examples_per_epoch('valid')
      num_valid_steps_per_epoch = -(-num_valid_examples // melt.eval_batch_size()) if num_valid_examples else None    
      logging.info('num_valid_examples:', num_valid_examples)

      if eval_fn:
        valid_ops = gezi.get_global('valid_ops')
        valid_iter = gezi.get_global('valid_tier')
        if not FLAGS.valid_hour:
          ofile = None
        else:
          odir = f'{FLAGS.log_dir}/infos/{FLAGS.valid_hour}' if not FLAGS.loop_fixed_valid else f'{FLAGS.log_dir}/infos/{FLAGS.train_hour}'
          os.system(f'mkdir -p {odir}')
          ofile = f'{odir}/valid.csv'
        metric_eval_fn = lambda model_path=None, write=write_valid: \
                                      evaluate(valid_ops, 
                                              valid_iter,
                                              num_steps=num_valid_steps_per_epoch,
                                              num_examples=num_valid_examples,
                                              eval_fn=eval_fn,
                                              names=valid_names,
                                              write_fn=valid_write_fn,
                                              model_path=model_path,
                                              write=write,
                                              num_gpus=num_gpus,
                                              suffix=valid_suffix,
                                              ofile=ofile,
                                              write_streaming=write_streaming,
                                              write_valid_only=FLAGS.write_valid_only,
                                              keys=eval_keys,
                                              sep=sep)
        if rank != 0:
          metric_eval_fn = None

    if FLAGS.test_input:
      num_test_examples = FLAGS.num_test or  Dataset.num_examples_per_epoch('test')
      num_test_steps_per_epoch = -(-num_test_examples // melt.eval_batch_size()) if num_test_examples else None
      logging.info('num_test_examples:', num_test_examples)
      test_ops = gezi.get_global('test_ops')
      test_iter = gezi.get_global('test_tier')
      inference_fn = lambda model_path=None: \
                              inference(test_ops, 
                                        test_iter,
                                        num_steps=num_test_steps_per_epoch,
                                        num_examples=num_test_examples,
                                        names=infer_names,
                                        debug_names=infer_debug_names,
                                        write_fn=infer_write_fn,
                                        model_path=model_path,
                                        num_gpus=num_gpus,
                                        suffix=infer_suffix,
                                        write_streaming=write_streaming,
                                        sep=sep)
      if rank != 0:
        inference_fn = None

  # if not FLAGS.valid_input:
  #   eval_ops = None

  train_flow(ops, 
             eval_ops=eval_ops,
             model_dir=FLAGS.model_dir,
             metric_eval_fn=metric_eval_fn,
             inference_fn=inference_fn,
             num_steps_per_epoch=num_steps_per_epoch,
             model=model,
             init_fn=init_fn,
             restore_fn=restore_fn,
             num_train_examples=num_examples,
             num_epochs=FLAGS.num_epochs,
             optimizer=optimizer,
             variables_list_fn=variables_list_fn,
             callbacks=callbacks,
             )
  return 0

# TODO support get test dataset
def get_datasets(Dataset):
  batch_size = melt.batch_size() if not FLAGS.use_horovod else FLAGS.batch_size
  valid_batch_size = melt.eval_batch_size() if not FLAGS.use_horovod else FLAGS.eval_batch_size
  # test_batch_size = valid_batch_size
  dataset = Dataset('train')
  FLAGS.num_train = len(dataset)
  dataset = dataset.make_batch(batch_size, repeat=True)

  eval_dataset = Dataset('valid')
  FLAGS.num_valid = len(eval_dataset)
  # eval_dataset = eval_dataset.make_batch(initializable=True)
  eval_dataset = eval_dataset.make_batch(valid_batch_size, initializable=True, repeat=False)
  # eval_dataset = eval_dataset.make_batch(initializable=True, repeat=True)

  valid_dataset = Dataset('valid')
  valid_dataset = valid_dataset.make_batch(valid_batch_size, repeat=True)

  return dataset, eval_dataset, valid_dataset

def get_train():
  melt.init()
  train = melt.eager.train if tf.executing_eagerly() or FLAGS.torch_only or tf.__version__ >= '2' else melt.apps.train
  return train

def get_fit():
  return get_train()

def fit(model, loss_fn=None, Dataset=None, dataset=None, eval_dataset=None, valid_dataset=None,
        num_folds=None, dry_run=False, **kwargs):
  fit_fn = get_fit()
  assert gezi.get('inited'), 'call melt.init at first'
  if FLAGS.keras:
    kwargs['return_info'] = True

  if 'eval_keys' in kwargs:
    FLAGS.eval_keys = kwargs['eval_keys']

  assert Dataset is not None or dataset is not None

  cv = False
  num_folds_ = FLAGS.num_folds
  ori_valid_input = FLAGS.valid_input

  dry_run = dry_run or FLAGS.dry_run or gezi.get('dry_run')

  if num_folds:
    FLAGS.num_folds = num_folds
  if FLAGS.num_folds:
    cv = True
  if cv:
    assert FLAGS.num_folds
    FLAGS.valid_input = None

  # if not (FLAGS.train_files and FLAGS.valid_files and len(FLAGS.train_files) == len(FLAGS.valid_files)):
  inputs = [FLAGS.train_input]
  valid_inputs = [FLAGS.valid_input]
  tests_inputs = []
  if FLAGS.train_input and ('|' in FLAGS.train_input or FLAGS.train_loop):
    inputs = [x for x in FLAGS.train_input.strip().split('|') if x]
    valid_span = FLAGS.valid_span
    if FLAGS.loop_type == 'day':
      if FLAGS.work_mode == 'train':
        valid_span = 1
    if FLAGS.train_only:
      valid_inputs = [None] * len(inputs)
    elif FLAGS.valid_span < 0:
      valid_inputs = [FLAGS.valid_input] * len(inputs) 
    elif valid_span == 0:
      valid_inputs = inputs
    else:
      FLAGS.valid_input = None
      inputs_ = inputs
      if FLAGS.ev_first:
        valid_span = 0
      if valid_span:
        valid_span = min(valid_span, len(inputs_) - 1)
        inputs = inputs_[:-valid_span]
        valid_inputs = inputs_[valid_span:] 
      if FLAGS.loop_train_all:
        inputs = inputs_
        if FLAGS.loop_fixed_valid:
          valid_inputs = [valid_inputs[-1]] * len(inputs)
        else:
          if valid_span:
            valid_inputs += [valid_inputs[-1]] * (len(inputs) - len(valid_inputs))
        # valid_inputs[-1] = None

    # else:
    #   inputs = FLAGS.train_files
    #   valid_inputs = FLAGS.valid_files
    #   tests_inputs = []
  
  if not inputs:
    logging.warning('Not enough to train and valid, exit -2')
    exit(-2)
  
  try:
    start_train_name = os.path.basename(inputs[0].split(',')[0].strip())
    end_valid_name = os.path.basename(valid_inputs[-1].split(',')[-1].strip()) if valid_inputs[-1] else None
    FLAGS.num_rounds = len(inputs) if not FLAGS.num_rounds else min(FLAGS.num_rounds, len(inputs))
    inputs = inputs[:FLAGS.num_rounds]
    valid_inputs = valid_inputs[:FLAGS.num_rounds]
    train_name = start_train_name
  except Exception:
    inputs = ['None']
    if valid_inputs:
      valid_inputs = ['None'] 
    train_name = 'None'
    pass

  if FLAGS.train_loop:
    if FLAGS.ev_first:
      FLAGS.ev_last = False
      FLAGS.use_valid_loss = False

    # actually eval_round means total rounds have trained until now, eval_round might not be set in melt.init by FLAGS.start_hour
    if FLAGS.eval_round is None:
      eval_round = melt.get_train_step()
      # # might be 0 as if no train_step.txt
      # if FLAGS.start_hour:
      #   eval_round2 = gezi.diff_hours(train_name, FLAGS.start_hour)
      #   ## 因为有的路径可能无效 所以实际eval_round2可能大于eval_round
      #   # assert not eval_round or eval_round == eval_round2, f'{eval_round} {eval_round2} {train_name}, {FLAGS.start_hour}'
      #   # assert eval_round2 >= eval_round, f'{eval_round2} {eval_round}'
      FLAGS.eval_round = eval_round

    try:
      if FLAGS.eval_day_step is None:
        FLAGS.eval_day_step = math.ceil(gezi.diff_days(train_name, FLAGS.start_hour, offset=1))
    except Exception:
      pass
  
  last_valid = -1
  # for i, (input_, valid_input_) in tqdm(enumerate(zip(inputs, valid_inputs)), total=FLAGS.num_rounds, desc='Round-%s' % train_name, ascii=False):
  idx = FLAGS.num_loop_dirs - max(FLAGS.valid_span, 1) if FLAGS.num_loop_dirs else len(inputs) - 1
  for i, (input_, valid_input_) in enumerate(zip(inputs, valid_inputs)):
    # if FLAGS.loop_range:
    if i < idx:
      FLAGS.do_test = False
    else:
      if FLAGS.loop_train:
        if not FLAGS.do_valid_last:
          FLAGS.do_valid = False
        FLAGS.do_test = True

    # if i < len(inputs) - FLAGS.test_span:
    #   FLAGS.do_test = False
    # else:
    #   if FLAGS.loop_train:
    #     if input_ == valid_input_ or FLAGS.valid_span < 1:
    #       FLAGS.do_test = True
    #       if not FLAGS.do_valid_last:
    #         FLAGS.do_valid = False
    #     else:
    #       # FLAGS.do_test = False
    #       FLAGS.do_valid = True

    # print(i, inputs, FLAGS.do_valid, FLAGS.do_test)

    input_ = input_.rstrip('/')
    gezi.set_global('start_time', time.time())
    if FLAGS.train_loop:
      if FLAGS.round == FLAGS.num_rounds - 1 and FLAGS.sync_valid_hour_final:
        FLAGS.sync_valid_hour = True
    if valid_input_ and ',' in valid_input_:
      # only first hour to be used for validation
      index = 0 if FLAGS.loop_type == 'hour' else FLAGS.valid_span - 1 
      valid_input_ = valid_input_.split(',')[index] 
    FLAGS.train_input = input_
    if FLAGS.dataset_rate >= 2:
      FLAGS.train_input = ','.join([FLAGS.train_input] * int(FLAGS.dataset_rate))
    FLAGS.valid_input = valid_input_
    if FLAGS.work_mode == 'valid' and not FLAGS.valid_input:
      FLAGS.valid_input = FLAGS.train_input

    if FLAGS.use_all_data:
      # TODO hack here
      others = [FLAGS.train_input.replace(FLAGS.base_data_name, other) for other in FLAGS.other_data_names.split(',')]
      data_other = ','.join([x for x in others if os.path.isdir(x.split(',')[0])])
      if data_other:
        FLAGS.train_input = f'{FLAGS.train_input},{data_other}'

    if FLAGS.use_all_type:
      src = 'tuwen' if 'tuwen' in FLAGS.train_input else 'video'
      dest = 'video' if src is 'tuwen' else 'tuwen'
      type_other = FLAGS.train_input.replace(src, dest)
      FLAGS.train_input = f'{FLAGS.train_input},{type_other}'

    if FLAGS.valid_use_all_data and FLAGS.valid_input:
      others = [FLAGS.valid_input.replace(FLAGS.base_data_name, other) for other in FLAGS.other_data_names.split(',')]
      data_other = ','.join([x for x in others if os.path.isdir(x)])
      if data_other:
        FLAGS.valid_input = f'{FLAGS.valid_input},{data_other}'

    train_name = os.path.basename(input_.split(',')[-1].strip())
    valid_name = os.path.basename(valid_input_) if valid_input_ else train_name

    FLAGS.valid_hour = valid_name
    if FLAGS.valid_hour and FLAGS.round >= FLAGS.valid_first_n and i != len(inputs) - 1:
      if FLAGS.valid_every_n and i % FLAGS.valid_every_n != 0:
        FLAGS.valid_input = None
      if FLAGS.valid_every_hash_n and gezi.hash(FLAGS.valid_hour) % FLAGS.valid_every_hash_n != 0:
        FLAGS.valid_input = None
      if FLAGS.novalid_max_n and (i - last_valid) > FLAGS.novalid_max_n:
        FLAGS.valid_inut = valid_input_

    if FLAGS.valid_input:
      last_valid = i

    FLAGS.version = FLAGS.valid_hour
      
    FLAGS.train_hour = train_name
    
    num_train_dirs = len(FLAGS.train_input.split(',')) if FLAGS.work_mode == 'train' else 0
    if FLAGS.work_mode != 'train':
      train_name = None
    logging.info('Round:', FLAGS.round, 'mode:', FLAGS.mode, \
        f'train_input:[{train_name}]', f'valid_input:[{valid_name}]', f'train_dirs:[{num_train_dirs}]', 'valid_dir:', FLAGS.valid_input, 'do_valid:', FLAGS.do_valid, 'do_test:', FLAGS.do_test)
  
    if FLAGS.loop_type == 'day':
      logging.info('Round:', FLAGS.round, \
        [os.path.basename(x) for x in input_.split(',')])

    if dry_run or FLAGS.hack_run:
      if FLAGS.loop_train and FLAGS.loop_type != 'day':
        logging.info('Round:', FLAGS.round, \
          [os.path.basename(x) for x in input_.split(',')])
      # FLAGS.round += 1
      # continue
      
    if FLAGS.loop_train:
      logging.info('--start_hour=%s' % start_train_name, '--end_hour=%s' % end_valid_name, 'root:', os.path.dirname(input_.split(',')[0]))
    assert FLAGS.train_only or not FLAGS.train_loop or FLAGS.work_mode != 'train' or FLAGS.ev_first or FLAGS.train_hour != FLAGS.valid_hour or FLAGS.loop_train_all

    FLAGS.valid_input = ori_valid_input 
    if FLAGS.work_mode in ['test', 'infer']:
      FLAGS.test_input = FLAGS.valid_input if not FLAGS.test_input else FLAGS.test_input

    if FLAGS.start_fold is None:
      folds = [None] if not cv else range(FLAGS.num_folds)
    else:
      folds = [FLAGS.start_fold] 
    if cv:
      assert FLAGS.keras, 'cv only suppport keras mode now'
      df = pd.DataFrame()
      save_model = FLAGS.save_model
      FLAGS.save_model = False
      ys = []
      preds = []
      xs = {}
      others = {}
      # model_weights = f'{FLAGS.model_dir}/model.h5'  
      # model.save_weights(model_weights)
      # model_path =  f'{FLAGS.model_dir}/model.h5'  
      # model.save(model_path)
      # model_path =  f'{FLAGS.model_dir}/model'
      # model.save(model_path, save_format='tf')
    for fold in folds:
      if fold is not None:
        logging.info(f'CrossValidation: ---------------------------: [{fold}/{FLAGS.num_folds}]')
        FLAGS.fold = fold      
        gezi.set('optimizer', None)

      info = fit_fn(model, loss_fn, Dataset, dataset, eval_dataset, valid_dataset, dry_run=dry_run, **kwargs)
      # must after fit_fn as where we got dataset input files and can do model(example) to infer shapes
      if fold == 0:
        init_weights = None
        if not FLAGS.cv_valid_only and tf.train.latest_checkpoint(FLAGS.ckpt_dir or FLAGS.model_dir) is None:
          # if not FLAGS.cv_save_temp:
          #   init_weights = model.get_weights()
          # else:
          os.system(f'mkdir -p {FLAGS.model_dir}/cv')
          model.save_weights(f'{FLAGS.model_dir}/cv/tmp.h5')

      gezi.set('info', info)
      if FLAGS.keras:
        import husky
        history = husky.train(model, loss_fn, info, Dataset, dry_run=dry_run, **kwargs)
      if info is not None:
        if FLAGS.allow_round_grow:
          FLAGS.round += 1
      if fold is not None:
        df = df.append(gezi.get('metrics'), ignore_index=True)
        # TODO might not ok for subclass api (see bert example XlmModel not ok, xlm_model ok)
        if not FLAGS.cv_valid_only:
          if FLAGS.cv_save_weights:
            os.system(f'mkdir -p {FLAGS.model_dir}/cv')
            model.save_weights(f'{FLAGS.model_dir}/cv/model_{fold}.h5')
          # if init_weights is not None:
          #   model.set_weights(init_weights)
          # else:
          if tf.train.latest_checkpoint(FLAGS.ckpt_dir or FLAGS.model_dir) is None:
            model.load_weights(f'{FLAGS.model_dir}/cv/tmp.h5')

        eval_callback = gezi.get('info')['eval_callback']
        ys += [eval_callback.y]
        preds += [eval_callback.preds]
        if eval_callback.x:
          if not xs:
            for key in eval_callback.x:
              xs[key] = [eval_callback.x[key]]
          else:
            for key in eval_callback.x:
              xs[key] += [eval_callback.x[key]]
        if eval_callback.other:
          if not others:
            for key in eval_callback.other:
              others[key] = [eval_callback.other[key]]
          else:
            for key in eval_callback.other:
              others[key] += [eval_callback.other[key]]       

        # except Exception:
        #   logging.warning(traceback.format_exc())
        #   logging.warning('TODO subclass api might has problem here')
        #   pass
        # model.load_weights(model_weights)
        # model = tf.keras.models.load_model(model_path)
        # model = tf.keras.experimental.load_from_saved_model(model_path)
    if cv and len(folds) > 1:
      FLAGS.save_model = save_model
      # results = dict(df.mean()[list(gezi.get('metrics').keys())].items())
      # gezi.pprint(results, print_fn=logging.info, desc='cv_metrics_mean:')
      # step = gezi.get('eval_step')
      # step += 1
      # # gezi.write_summaries(results, step)
      # eval_callback.logger.scalars(results, step)
      # eval_callback.writer.write(results)
      if ys and ys[0] is not None and len(ys[0]):
        # TODO work for using eval(dataset)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        if xs:
          for key in xs:
            xs[key] = np.concatenate(xs[key])
        if others:
          for key in others:
            others[key] = np.concatenate(others[key])
        kwargs = {}
        args = inspect.getargspec(eval_callback.eval_fn).args    
        if 'info' in args:
          kwargs['info'] = xs
        if 'x' in args:
          kwargs['x'] = xs
        if 'model' in args:
          kwargs['model'] = model
        if 'other' in args:
          kwargs['other'] = others
        results = eval_callback.eval_fn(ys, preds, **kwargs)
        # logging.info(f'cv_metrics: {list(zip(names, vals_))}  num examples: {len(ys)}')
        gezi.pprint(results, print_fn=logging.info, desc='cv_metrics_full:')
        step = gezi.get('eval_step')
        step += 1
        # gezi.write_summaries(results, step)
        if eval_callback.logger is not None:
          eval_callback.logger.scalars(results, step)
        eval_callback.writer.write(results)
        gezi.set('eval_step', step)

      vdfs, tdfs = [], []
      for i in range(FLAGS.folds):
        vfile = f'{FLAGS.model_dir}/infos/valid_{i}.csv'
        if os.path.exists(vfile):
          vdfs.append(pd.read_csv(vfile))
        tfile = f'{FLAGS.model_dir}/infos/test_{i}.csv'
        if os.path.exists(tfile):
          tdfs.append(pd.read_csv(tfile))
      
      if vdfs:
        assert len(vdfs) == FLAGS.folds, len(vdfs)
        df = pd.concat(vdfs)
        df.to_csv(f'{FLAGS.model_dir}/cv/valid.csv', index=False)
      if tdfs:
        assert len(tdfs) == FLAGS.folds, len(tdfs)
        df = pd.concat(tdfs)
        df = df.groupby('id', as_index=False).mean()
        df.to_csv(f'{FLAGS.model_dir}/cv/submission.csv', index=False)
        os.system(f'cp -rf {FLAGS.model_dir}/infos/test*.csv {FLAGS.model_dir}/cv')

    FLAGS.num_folds = num_folds_
    
    melt.save_eval_step()  

  # if FLAGS.wandb_key:
  #   try:
  #     import wandb
  #     wandb.finish()
  #   except Exception as e:
  #     logging.warning(e)
  os.system(f'touch {FLAGS.model_dir}/done.txt')
  logging.info(f'ALL DONE! You may check log file: [tail {FLAGS.log_dir}/log.html*]')
