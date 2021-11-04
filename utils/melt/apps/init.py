#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   init.py
#        \author   chenghuige
#          \date   2019-09-07 11:04:41.961977
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fcntl
import gezi
import melt
from melt.apps.config import *
import sys
import os
import glob
import traceback
import time, timeit
from datetime import datetime, timedelta
from tqdm import tqdm
import random
import numpy as np
import psutil
import json
import pickle
from absl.testing import flagsaver

import tensorflow as tf
from tensorflow import keras
from absl import flags

FLAGS = flags.FLAGS

# import melt.utils.logging as logging
logging = gezi.logging

import torch


def is_inited():
  return gezi.get('inited')


def reset():
  gezi.set('inited', False)


def reinit():
  gezi.set('inited', False)
  init()


def init_flags():
  from absl import flags
  FLAGS = flags.FLAGS
  FLAGS([''])


def get_flags():
  from absl import flags
  FLAGS = flags.FLAGS
  return FLAGS


def _restart(allow_cpu=False):
  assert FLAGS.num_gpus != 0
  command = FLAGS.command
  gpus = gezi.get_gpus(FLAGS.min_free_gpu_mem,
                       FLAGS.max_used_gpu_mem,
                       env_limit=False)
  num_gpus = FLAGS.num_gpus if FLAGS.num_gpus else 1
  if FLAGS.mode == 'async_valid':
    num_gpus = 1
  if ((len(gpus) < num_gpus) or FLAGS.cpu_valid or
      gezi.get_env('CUDA_VISIBLE_DEVICES') == '-1') and (allow_cpu and
                                                         FLAGS.allow_cpu):
    logging.info('restart as CUDA_VISIBLE_DEVICES=-1')
    command = f'CUDA_VISIBLE_DEVICES=-1 {command}'
  elif FLAGS.num_valid_gpus > 1:
    # TODO 也许需要非horovod模式多卡并行验证
    command = f'horovodrun -np {FLAGS.num_valid_gpus} {command}'
  elif 'CUDA_VISIBLE_DEVICES' in os.environ or (FLAGS.torch and
                                                not FLAGS.torch_only):
    # Mostly for torch using eager read tfrecord currently set device not work so need to force using 1 gpu
    if len(gpus) < num_gpus:
      gpus = gezi.get_gpus(None, None, env_limit=False)
    if not gpus:
      gpus = list(range(num_gpus))
    else:
      gpus = gpus[:num_gpus]
    gpus_str = ','.join(map(str, gpus[:num_gpus]))
    logging.info(f'restart as CUDA_VISIBLE_DEVICES={gpus_str}')
    command = f'CUDA_VISIBLE_DEVICES={gpus_str} {command}'
  if not command.endswith('&'):
    command += ' &'
  gezi.system(command)
  exit(0)


def _get_gpus(num_gpus=None, time_out=None, exit_fn=None):
  gpus = gezi.get_gpus(FLAGS.min_free_gpu_mem,
                       FLAGS.max_used_gpu_mem,
                       env_limit=True)

  if not num_gpus:
    return gpus

  if not FLAGS.wait_gpu_span:
    assert len(
        gpus
    ) >= num_gpus, f'Not enough gpus with min_free_gpu_mem >= {FLAGS.min_free_gpu_mem}Mb and max_used_gpu_mem <= {FLAGS.max_used_gpu_mem}, try to use {num_gpus} but only has {len(gpus)} {gpus} CUDA_VISIABLE_DEVICES={gezi.get_specific_gpus()} mode:{FLAGS.mode} work_mode:{FLAGS.work_mode}'
  else:
    used_time = 0
    while len(gpus) < num_gpus:
      if time_out and used_time >= time_out:
        assert exit_fn
        exit_fn(len(gpus))
      logging.warning(
          f'Not enough gpus with min_free_gpu_mem >= {FLAGS.min_free_gpu_mem}Mb and max_used_gpu_mem <= {FLAGS.max_used_gpu_mem}, try to use {num_gpus} but only has {len(gpus)} {gpus} CUDA_VISIABLE_DEVICES={gezi.get_specific_gpus()} mode:{FLAGS.mode}, work_mode:{FLAGS.work_mode}'
      )
      time.sleep(FLAGS.wait_gpu_span)
      gpus = gezi.get_gpus(FLAGS.min_free_gpu_mem,
                           FLAGS.max_used_gpu_mem,
                           env_limit=True)
      used_time += FLAGS.wait_gpu_span

  return gpus[:num_gpus]


def _check_tpu():
  tpu = gezi.get('tpu')
  if tpu:
    return tpu
  try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Running on TPU ', tpu.master())
  except ValueError:
    tpu = None

  if tpu:
    gezi.set('tpu', tpu)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    try:
      strategy = tf.distribute.TPUStrategy(tpu)
    except Exception:
      strategy = tf.distribute.experimental.TPUStrategy(tpu)
    melt.distributed.set_strategy(strategy)
    FLAGS.num_gpus = strategy.num_replicas_in_sync

  return tpu


def get_tpu():
  return _check_tpu()


def setup_tpu():
  return _check_tpu()


def _set_strategy():
  if FLAGS.num_gpus > 1 and not FLAGS.torch:
    if not FLAGS.distribute_strategy:
      FLAGS.distribute_strategy = 'mirrored'
    melt.distributed.set_strategy(FLAGS.distribute_strategy, FLAGS.num_gpus)


# TODO split init function


def init(graph=None):
  if gezi.get('inited'):
    return

  timer = gezi.Timer()
  start_time = timeit.default_timer()
  gezi.set('start_time', start_time)

  if not FLAGS.icecream:
    from icecream import ic
    ic.disable()

  if not FLAGS.icecream_context:
    ic.configureOutput(includeContext=False)

  if FLAGS.print_version:
    import torch
    from icecream import ic
    logging.debug('tf', tf.__version__, 'torch', torch.__version__)
    ic(tf.__version__, torch.__version__)
    
  dist = gezi.DistributedWrapper()
  distributed = dist.is_distributed()
  comm = dist.comm

  # Notice for horovod run, each worker should has seem FLAGS.seed
  if 'SEED' in os.environ:
    FLAGS.seed = int(os.environ['SEED'])

  rand_seed = False
  if FLAGS.seed is None:
    FLAGS.seed = random.randint(0, 100000)
    rand_seed = True
  elif FLAGS.seed == 0:
    FLAGS.seed = None

  # each worker should use same seed, TODO for torch ?
  if rand_seed:
    if FLAGS.use_horovod:
      temp = np.array([FLAGS.seed])
      comm.Bcast(temp, root=0)
      FLAGS.seed = temp[0]
    elif distributed:
      # 单独程序验证ok 但是这里跑会hang FIXME 当前采用mruntorch.py传递--seed
      l = [torch.as_tensor(FLAGS.seed).to('cuda')]
      dist.dist.broadcast_multigpu(l, 0)
      FLAGS.seed = int(l[0].cpu().numpy())
      dist.dist.barrier()

  if FLAGS.seed:
    tf.random.set_seed(FLAGS.seed)

  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  np.set_printoptions(precision=FLAGS.print_precision)

  if FLAGS.gradient_accumulation_steps > 1:
    FLAGS.batch_size = int(FLAGS.batch_size / FLAGS.gradient_accumulation_steps)

  ori_batch_size = FLAGS.base_batch_size or FLAGS.batch_size

  if FLAGS.debug:
    FLAGS.log_level = min(FLAGS.log_level, 10)
    gezi.set('debug', True)

  if FLAGS.mode == 'valid_perf':
    FLAGS.work_mode = 'valid'
  mode = FLAGS.mode or FLAGS.work_mode or 'train'

  if FLAGS.model_suffix:
    FLAGS.model_dir += FLAGS.model_suffix

  if FLAGS.model_name:
    if FLAGS.kaggle_prefix and os.path.exists(
        '/kaggle') and not FLAGS.model_name.endswith(
            '.kaggle') and FLAGS.mode != 'async_valid':
      FLAGS.model_name += '.kaggle'
    if os.path.exists(FLAGS.model_name):
      FLAGS.model_name = os.path.basename(FLAGS.model_name)

    FLAGS.model_dir = os.path.join(os.path.dirname(FLAGS.model_dir),
                                   FLAGS.model_name)
  if FLAGS.model_root:
    FLAGS.model_dir = os.path.join(
        FLAGS.model_root,
        os.path.basename(gezi.strip_suffix(FLAGS.model_dir, '/')))

  if FLAGS.model_dir is None:
    FLAGS.model_dir = '/tmp/melt'

  if FLAGS.model_name_suffix and FLAGS.mode != 'async_valid':
    FLAGS.model_name += FLAGS.model_name_suffix
    FLAGS.model_dir += FLAGS.model_name_suffix
    
  if FLAGS.model_seed_suffix and FLAGS.mode != 'async_valid':
    FLAGS.model_name += f'.s{FLAGS.seed}'
    FLAGS.model_dir += f'.s{FLAGS.seed}'

  if FLAGS.model_time_suffix and FLAGS.mode != 'async_valid':
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    FLAGS.model_name += f'.{dt_string}'
    FLAGS.model_dir += f'.{dt_string}'

  if not FLAGS.model_name and FLAGS.model_dir:
    FLAGS.model_name = os.path.basename(FLAGS.model_dir)

  if not FLAGS.log_dir:
    if 'log_dir' in os.environ and os.environ['log_dir']:
      FLAGS.log_dir = os.environ['log_dir']
  if not FLAGS.log_dir and FLAGS.model_dir:
    if os.path.isfile(FLAGS.model_dir):
      FLAGS.log_dir = os.path.dirname(FLAGS.model_dir)
    else:
      FLAGS.log_dir = FLAGS.model_dir

  model_exists = False
  if FLAGS.model_dir:
    if os.path.exists(FLAGS.model_dir):
      model_exists = True
    if os.path.isfile(FLAGS.model_dir + '.index'):
      FLAGS.log_dir = os.path.dirname(FLAGS.model_dir)

  if FLAGS.log_dir_under_model_dir:
    FLAGS.log_dir = os.path.join(FLAGS.model_dir, 'log')

  assert not (FLAGS.check_exists and model_exists and mode == 'train')

  rlog_dir = os.path.realpath(FLAGS.log_dir)
  # if FLAGS.hdfs_mark in rlog_dir:
  #     FLAGS.log_dir = rlog_dir.replace(FLAGS.hdfs_mark, FLAGS.local_mark)
  rmodel_dir = os.path.realpath(FLAGS.model_dir)
  # if FLAGS.hdfs_mark in rmodel_dir:
  #     FLAGS.model_dir = rmodel_dir.replace(FLAGS.hdfs_mark, FLAGS.local_mark)
  FLAGS.ori_log_dir = FLAGS.log_dir if not FLAGS.ori_log_dir or FLAGS.hdfs_mark in rlog_dir else os.path.join(
      FLAGS.ori_log_dir, os.path.basename(FLAGS.log_dir))
  gezi.system(f'mkdir -p {FLAGS.ori_log_dir}')
  FLAGS.ori_model_dir = FLAGS.model_dir if not FLAGS.ori_model_dir or FLAGS.hdfs_mark in rmodel_dir else os.path.join(
      FLAGS.ori_model_dir, os.path.basename(FLAGS.model_dir))
  gezi.system(f'mkdir -p {FLAGS.ori_model_dir}')

  if mode != 'train' and (not FLAGS.pretrain):
    FLAGS.clear_first = False

  if FLAGS.dry_run:
    FLAGS.clear_first = False

  if FLAGS.clear:
    FLAGS.clear_first = True

  if FLAGS.clear_first and FLAGS.model_dir and os.path.exists(FLAGS.model_dir):
    logging.info('Removing model dir', FLAGS.model_dir)
    os.system('rm -rf %s/*' % FLAGS.model_dir)

  if FLAGS.clear_first and FLAGS.log_dir and os.path.exists(
      FLAGS.log_dir) and FLAGS.log_dir != FLAGS.model_dir:
    logging.info('Removing log dir', FLAGS.log_dir)
    os.system('rm -rf %s/*' % FLAGS.log_dir)

  if FLAGS.clear_first and FLAGS.ori_model_dir and os.path.exists(
      FLAGS.ori_model_dir):
    logging.info('Removing ori model dir', FLAGS.ori_model_dir)
    os.system('rm -rf %s/*' % FLAGS.ori_model_dir)

  if FLAGS.clear_first and FLAGS.ori_log_dir and os.path.exists(
      FLAGS.ori_log_dir) and FLAGS.ori_log_dir != FLAGS.ori_model_dir:
    logging.info('Removing ori log dir', FLAGS.ori_log_dir)
    os.system('rm -rf %s/*' % FLAGS.ori_log_dir)

  if FLAGS.clear:
    exit(0)

  # HACK for publicData/CloudS which is hdfs not work with writing file during train
  # if log is not same as model dir then will sync later
  # TODO remove this if python logging bug fixe for CloudS
  # assert not FLAGS.hdfs_mark in rlog_dir, 'CloudS not support write now'
  if FLAGS.hdfs_mark in rlog_dir:
    gezi.system(f'mkdir -p {rlog_dir}')
    # command = f"rsync -avP --ignore-existing --update --exclude 'models*' {rlog_dir} {os.path.dirname(FLAGS.log_dir)}"
    # gezi.system(command)

  if FLAGS.hdfs_mark in rmodel_dir:
    FLAGS.model_dir = rmodel_dir.replace(FLAGS.hdfs_mark, FLAGS.local_mark)
    gezi.system(f'mkdir -p {rmodel_dir}')
    # if FLAGS.model_dir != FLAGS.log_dir:
    #     command = f"rsync -avP --ignore-existing --update --exclude 'models*' {rmodel_dir} {os.path.dirname(FLAGS.model_dir)}"
    #     gezi.system(command)

  os.system('mkdir -p %s' % FLAGS.log_dir)
  os.system('mkdir -p %s' % FLAGS.model_dir)

  if FLAGS.gcs_root and FLAGS.wandb_id:
    FLAGS.gcs_src = f'{FLAGS.gcs_root}/{FLAGS.wandb_id}'

  if FLAGS.gcs_src:
    gezi.system(f'gsutil -m cp -r {FLAGS.gcs_src}/* {FLAGS.model_dir}')

  if FLAGS.restore_configs:
    gezi.restore_configs(
        FLAGS.pretrain or FLAGS.log_dir or FLAGS.model_dir,
        ignore_names=[
            'mode', 'work_mode', 'num_gpus', 'log_dir', 'model_dir',
            'model_name', 'pretrained_dir', 'pretrained', 'pretrain',
            'load_weights_only', 'load_by_name', 'load_skip_mismatchs',
            'batch_size', 'eval_batch_size', 'fp16', 'fold', 'folds', 'input',
            'train_input', 'valid_input', 'test_input', 'train_exclude',
            'valid_exclude', 'train_only'
        ])
  flag_values = flagsaver.save_flag_values()

  # assert FLAGS.log_dir, 'you need to set log_dir or model_dir'
  if FLAGS.verbose is not None:
    FLAGS.quiet = not FLAGS.verbose

  logtofile = True if FLAGS.mode != 'async_valid' else False
  logging.init(FLAGS.log_dir,
               quiet=FLAGS.quiet,
               level=FLAGS.log_level,
               logtofile=logtofile)
  logging.debug('tf version:', tf.__version__)
  import torch
  logging.debug('torch version:', torch.__version__)
  logging.debug('seed', FLAGS.seed)  

  tpu = _check_tpu()

  if tpu is None:
    #FLAGS.drop_remainder = FLAGS.drop_remainder or False
    pass
  else:
    #FLAGS.drop_remainder = FLAGS.drop_remainder or True
    FLAGS.async_valid = False  # only use async_valid in multiple gpu enviroment

  if FLAGS.masked_fields:
    FLAGS.masked_fields = FLAGS.masked_fields.strip("'")

  if FLAGS.save_only:
    FLAGS.steps = -1

  if FLAGS.profiling:
    FLAGS.enable_profiling = True
    if not FLAGS.profile_interval_steps:
      FLAGS.profile_interval_steps = 100

  if mode == 'show':
    FLAGS.plot_graph = True

  if FLAGS.plot_graph or mode != 'train' or FLAGS.num_steps < 0:
    FLAGS.rounds = 1

  if FLAGS.plot_graph:
    FLAGS.keras = False
    FLAGS.eager = True

  FLAGS.days = FLAGS.days or FLAGS.num_days
  FLAGS.hours = FLAGS.hours or FLAGS.num_hours

  if FLAGS.num_eval_steps:
    FLAGS.vie = FLAGS.num_epochs / FLAGS.num_eval_steps
  # if FLAGS.interval_steps > FLAGS.valid_interval_steps:
  #     FLAGS.valid_interval_steps = FLAGS.interval_steps
  # assert not FLAGS.interval_steps or FLAGS.valid_interval_steps % FLAGS.interval_steps == 0

  ## dynamic_pad dpreciated
  #if FLAGS.padded_tfrecord or not FLAGS.sparse_to_dense:
  #    FLAGS.dynamic_pad = False

  #if not FLAGS.batch_parse and FLAGS.sparse_to_dense:
  #    FLAGS.dynamic_pad = True

  if FLAGS.batch_sizes:
    FLAGS.batch_sizes = [int(x) for x in FLAGS.batch_sizes]
    FLAGS.batch_size = FLAGS.batch_sizes[0]

  if FLAGS.torch_only or gezi.get_env('TORCH') == '1':
    FLAGS.torch = True

  # if not FLAGS.torch:
  #     # for tf if not set we will let one process only use 1 gpu (empty gpu will mostly use 10M so here for safe set 20M) Here move to outer script
  #     if FLAGS.max_used_gpu_mem is None:
  #       FLAGS.max_used_gpu_mem = 20

  if FLAGS.torch:
    FLAGS.allow_growth = True

  if FLAGS.job_name:
    FLAGS.ps_strategy = True

  if FLAGS.ps_strategy and FLAGS.ps_hosts:
    if FLAGS.num_gpus and FLAGS.num_gpus > 1 and not FLAGS.worker_hosts:
      host, port = FLAGS.ps_hosts.split(',')[-1].split(':')
      port = int(port)
      port += 1
      worker_hosts = []

      # localhost:8830
      for i in range(FLAGS.num_gpus):
        worker_hosts.append(f'{host}:{port + i}')

      FLAGS.worker_hosts = ','.join(worker_hosts)
      FLAGS.num_gpus = 1

  FLAGS.distributed = distributed
  FLAGS.local_rank = dist.local_rank()
  FLAGS.world_size = dist.size()
  gezi.set_global('dist', dist)
  # TODO mpi means horovod ? better handle
  if dist.is_hvd:
    FLAGS.use_horovod = True

  if FLAGS.use_horovod:
    FLAGS.min_free_gpu_mem = None

  if FLAGS.debug_model:
    gezi.set('debug_model', True)
    gezi.set('model_summary', True)

  if FLAGS.model_summary:
    gezi.set('model_summary', True)


  if 'FOLD' in os.environ:
    try:
      FLAGS.fold = int(os.environ['FOLD'])
    except Exception:
      sub_dir = os.environ['FOLD']
      assert os.path.isdir(FLAGS.model_dir)
      FLAGS.model_dir = os.path.join(FLAGS.model_dir, sub_dir)
      pass
  if FLAGS.fold == -1:
    FLAGS.fold = None
  # if FLAGS.fold is not None:
  #     FLAGS.valid_input = None

  FLAGS.command = ' '.join(['python'] + sys.argv)
  logging.debug('input command:\n', FLAGS.command)

  num_available_gpus = gezi.get_num_available_gpus()

  if num_available_gpus == 1 and not FLAGS.allow_cpu:
    FLAGS.async_valid = False

  if FLAGS.num_gpus != None and (num_available_gpus < FLAGS.num_gpus or
                                 FLAGS.num_gpus < 0) and tpu is None:
    FLAGS.num_gpus = num_available_gpus

  # ps_strategy means tf distributed..
  if distributed and tpu is None and not FLAGS.ps_strategy:
    FLAGS.num_gpus = 0

  if FLAGS.learning_rates and len(FLAGS.learning_rates) > 1:
    # l = np.fromstring(FLAGS.learning_rates, dtype=float, sep=',')
    FLAGS.learning_rates = [float(x) for x in FLAGS.learning_rates]
    FLAGS.learning_rate, FLAGS.learning_rate2 = FLAGS.learning_rates[
        0], FLAGS.learning_rates[1]
    if FLAGS.num_learning_rates is None:
      FLAGS.num_learning_rates = len(FLAGS.learning_rates)
  else:
    FLAGS.num_learning_rates = 1
    FLAGS.learning_rates = [FLAGS.learning_rate]

  if FLAGS.optimizers:
    l = FLAGS.optimizers.split(',')
    if len(l) > 1:
      FLAGS.optimizer, FLAGS.optimizer2 = l[0], l[1]
    else:
      FLAGS.optimizer = l[0]
    if FLAGS.num_optimizers is None:
      FLAGS.num_optimizers = len(l)
  else:
    FLAGS.num_optimizers = 1

  if not FLAGS.bert_style_lr:
    if FLAGS.optimizer:
      FLAGS.optimizer = FLAGS.optimizer.replace('bert-', '')
    if FLAGS.optimizer2:
      FLAGS.optimizer2 = FLAGS.optimizer2.replace('bert-', '')

  FLAGS.num_learning_rates = min(FLAGS.num_learning_rates, FLAGS.num_optimizers)

  if 'FIXED_BATCH_SIZE' in os.environ and os.environ['FIXED_BATCH_SIZE'] == '1':
    FLAGS.batch_size_per_gpu = False
  if 'BATCH_SIZE_PER_GPU' in os.environ:
    if os.environ['BATCH_SIZE_PER_GPU'] == '0':
      FLAGS.batch_size_per_gpu = False
    elif os.environ['BATCH_SIZE_PER_GPU'] == '1':
      FLAGS.batch_size_per_gpu = True

  # 用户显示的 用 CUDA_VISIBLE_DEVICES=设置的 gpu数目
  # 如果用户没有设置 sh ./train.sh 比如有8个gpu 理论都可用 tf的话按照 num_gpus=1处理 只用1个 如果想用多个 必须显示的指定 CUDA_VISIABLE_DEVICES 或者设置 num_gpus
  # torch 本来也可以复用这个逻辑 但是实际操作的时候 发现torch会始终占用这个8个 无法内部再限制操作 因此torch 的逻辑是有多少可用gpu就按几个gpu处理
  # 如果想只用1个 那么必须显示的设置 CUDA_VISIBLE_DEVICES= 来限制
  num_specific_gpus = gezi.get_num_specific_gpus()

  logging.debug('num_available_gpus:', num_available_gpus, 'num_specific_gpus:',
                num_specific_gpus)
  # if (FLAGS.torch and not FLAGS.torch_only) and not num_specific_gpus:
  #     logging.error('torch mode must set CUDA_VISIBLE_DEVICES restart')
  #     _restart(allow_cpu=False)

  if FLAGS.num_gpus is None:
    if not num_specific_gpus:
      logging.debug(
          'Not set CUDA_VISIBLE_DEVICES, by default try to only use 1 gpu')
    FLAGS.num_gpus = num_specific_gpus if num_specific_gpus else 1

  # TODO hack, melt.num_gpus() is num gpus without considering horovod, melt.num_gpus2() is turely num gpus used
  melt.set_global('num_gpus', max(FLAGS.num_gpus, 1))

  if not FLAGS.batch_size_per_gpu and FLAGS.num_gpus > 1:
    logging.info(
        'batch size is shrink by %d for each gpu to make total insts per step still %d'
        % (FLAGS.num_gpus, FLAGS.batch_size))
    FLAGS.batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)
    if FLAGS.eval_batch_size:
      FLAGS.eval_batch_size = int(FLAGS.eval_batch_size / FLAGS.num_gpus)

  gpus = []

  if FLAGS.torch or FLAGS.mode == 'async_valid':
    timeout = 10
  else:
    timeout = FLAGS.wait_gpu_timeout

  def _exit_fn(num_free_gpus):
    if FLAGS.mode == 'async_valid':
      logging.warning(
          'Valid:Timeout waiting gpu exit 0 and do async valid again')
      _restart(allow_cpu=FLAGS.allow_cpu)
      # _restart(allow_cpu=True if FLAGS.torch else False)
    else:
      logging.warning('Train:Timeout waiting gpu exit -1')
      exit(-1)

  ## init 等待排队 如果两个进程同时init抢占了gpu 会有一个可能OOM 但是supervise.py可以重启继续
  # # TODO FIXME still not safe might two run access same gpu whith later one OOM.. Might due to after init still first run not use enough gpu mem..
  fp = None
  lock_file = os.path.join(os.environ['HOME'], FLAGS.lock_file)

  if not FLAGS.log_distributed:
    logging.set_dist(dist)

  if distributed:
    # 当ps_strategy 不算在distributed
    logging.info('distributed', distributed, dist.dist)

    if dist.rank() == 0:
      fp = open(lock_file, 'w')
      logging.info(
          'fcntl.floc with lock_file', lock_file,
          '(If hang here means other programs calling melt.init have not finished yet)'
      )
      fcntl.flock(fp, fcntl.LOCK_EX)

    if gezi.get_env('CUDA_VISIBLE_DEVICES') != '-1':
      gpus = _get_gpus(dist.size() if not FLAGS.ps_strategy else FLAGS.num_gpus,
                       timeout, _exit_fn)

    logging.info('In distributed mode using %d gpus' % dist.size())

    if FLAGS.distributed_scale:
      if not FLAGS.learning_rates:
        FLAGS.learning_rate = FLAGS.learning_rate * max(dist.size(), 1)
      else:
        FLAGS.learning_rates = ','.join([
            str(int(x) * dist.size()) for x in FLAGS.learning_rates.split(',')
        ])
      logging.info('using distributed multipy learning rate by {} to {}'.format(
          dist.size(), 'lrs:', FLAGS.learning_rates, 'lr:',
          FLAGS.learning_rate))
  else:
    if gezi.get_env('CUDA_VISIBLE_DEVICES') != '-1':
      fp = open(lock_file, 'w')
      logging.info(
          'fcntl.floc with lock_file', lock_file,
          '(If hang here means other programs calling melt.init have not finished yet)'
      )
      fcntl.flock(fp, fcntl.LOCK_EX)
      gpus = _get_gpus(FLAGS.num_gpus, timeout, _exit_fn)

  gezi.set('fp', fp)

  if FLAGS.buckets:
    assert FLAGS.length_key, 'must set length key if using buckets'
    if FLAGS.batch_parse:
      logging.warning('Buckets dataset force to set batch_parse to False')
      FLAGS.batch_parse = False

  if 'TPU' in os.environ and int(os.environ['TPU']) == 1:
    FLAGS.use_tpu = True

  if FLAGS.use_tpu:
    from tensorflow.contrib.tpu.python.tpu import tpu_function
    # Add this somewhere at the top
    tpu_function.get_tpu_context().set_number_of_shards(FLAGS.num_tpu_cores)

  if 'METRIC' in os.environ and os.environ['METRIC'] == 'epoch':
    FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'epoch')

  if 'METRIC' in os.environ and os.environ['METRIC'] == '1':
    FLAGS.work_mode = 'valid'

  if 'DOUBLE_BATCH' in os.environ:
    FLAGS.batch_size *= 2
    if FLAGS.batch_sizes:
      l = [str(int(x) * 2) for x in FLAGS.batch_sizes.split(',')]
      FLAGS.batch_sizes = ','.join(l)
    # FLAGS.model_dir = os.path.join(os.path.dirname(FLAGS.model_dir.rstrip('/')) + '/double', os.path.basename(FLAGS.model_dir.rstrip('/')))
  
  if FLAGS.batch_size_scale != 1:
    FLAGS.batch_size = int(FLAGS.batch_size * FLAGS.batch_size_scale)
    if FLAGS.batch_sizes:
      l = [str(int(x * FLAGS.batch_size_scale)) for x in FLAGS.batch_sizes.split(',')]
      FLAGS.batch_sizes = ','.join(l)

  if 'DOUBLE_LEN' in os.environ:
    if FLAGS.buckets:
      l = [str(int(x) * 2) for x in FLAGS.buckets.split(',')]
      FLAGS.buckets = ','.join(l)
    try:
      FLAGS.content_limit *= 2
    except Exception:
      pass

  if FLAGS.eval_batch_size:
    FLAGS.eval_batch_size = int(FLAGS.eval_batch_size * FLAGS.valid_multiplier)
  else:
    FLAGS.eval_batch_size = int(FLAGS.batch_size * FLAGS.valid_multiplier)

  # if FLAGS.train_scratch or 'SCRATCH' in os.environ:
  #     if os.path.exists(os.path.join(FLAGS.model_dir, 'log.html')):
  #         if not os.path.exists(os.path.join(FLAGS.model_dir, 'log.txt')) or gezi.get_unmodify_minutes(os.path.join(FLAGS.model_dir, 'log.txt')) < 10:
  #             logging.info('In scratch mode and found model_dir',
  #                          FLAGS.model_dir, 'exit(0)')
  #             exit(0)
  #         else:
  #             logging.info(
  #                 'In scratch mode but log.txt un modify for long time, continue running')
  if FLAGS.mode == 'eval':
    FLAGS.mode = 'valid'
  if FLAGS.mode is not None:
    if FLAGS.mode == 'async_valid':
      FLAGS.work_mode = 'valid'
    else:
      FLAGS.work_mode = FLAGS.mode

  if 'MODE' in os.environ:
    FLAGS.work_mode = os.environ['MODE']

  if FLAGS.work_mode != 'train':
    FLAGS.train_scratch = False

  if 'PYTORCH' in os.environ or 'PYT' in os.environ or 'TORCH' in os.environ:
    FLAGS.torch = True

  if FLAGS.torch:
    FLAGS.keras = False

  # 如果没有特别设定 tf2 默认keras模式
  if tf.__version__ < '2' and FLAGS.keras == None:
    FLAGS.keras = False

  if tf.__version__ >= '2' and FLAGS.keras == None:
    FLAGS.keras = True

  if FLAGS.work_mode == 'count':
    FLAGS.eager = True

  if FLAGS.work_mode != 'train':
    FLAGS.valid_span -= 1
    FLAGS.num_rounds = 1
    FLAGS.rounds = 1
    FLAGS.cache_valid = False

  if FLAGS.tf_v2:
    FLAGS.eager = True

  # if tf.__version__ >= '2' and not FLAGS.graph:
  #     FLAGS.eager = True

  # print(tf.__version__, tf.__version__ >= '2')

  if tf.__version__ >= '2':
    FLAGS.eager = True
    # tf.config.experimental_run_functions_eagerly(True)
    # tf.config.run_functions_eagerly(True)

  if tf.__version__ >= '2' and FLAGS.graph and not FLAGS.torch:
    tf.compat.v1.disable_eager_execution()

  if FLAGS.torch and not FLAGS.torch_only:
    FLAGS.eager = True

  if FLAGS.torch:
    FLAGS.allow_growth = True
    FLAGS.allow_soft_placement = True

  if FLAGS.run_eagerly:
    tf.config.run_functions_eagerly(True)

  # -------------------------setup gpus
  gpus_global = gpus
  # --change to local gpu index
  specific_gpus = gezi.get_specific_gpus()
  if specific_gpus:
    m = dict([(x, i) for i, x in enumerate(specific_gpus)])
    gpus = [m[x] for x in gpus]

  # TODO torch mode should use tf eager mode reading but since some bug for gpu oom.. now just use graph mode
  if FLAGS.eager or 'EAGER' in os.environ and int(
      os.environ['EAGER']
  ) == 1 or 'SHOW' in os.environ or FLAGS.torch or tf.__version__ >= '2':
    # if torch and horovod can not use eager mode will conflict with torch.cuda.set_device, each process will use all gpus then
    # if not FLAGS.torch_only:
    logging.debug('-------------RUN IN EAGER MODE!')
    if FLAGS.torch:
      if tf.__version__ < '2':
        ## 有效 如果 sh ./bin 会只有1个gpu对torch可见 而且tf并不占用gpu空间
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        # config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = FLAGS.allow_growth
        config.allow_soft_placement = FLAGS.allow_soft_placement
        if distributed:
          gpus = [dist.local_rank()]
        config.gpu_options.visible_device_list = ','.join(map(str, gpus))
        config.gpu_options.per_process_gpu_memory_fraction = 0.
        tf.compat.v1.enable_eager_execution(config=config)
      else:
        ## TODO 这样运行过程中不占用gpu但是启动是所有可见gpu都分配了255M 如果sh ./bin  就是所有8个gpu都对torch可见,当然也可以这样强制配合CUDA_VISIBLE_DIVICE来使用
        # tf.config.set_visible_devices([], 'GPU')
        ## 下面做法类似上面tf1版本 只分配占用1个gpu 但是注意tf读取dataset使用了gpu分配了1g多空间给tf 另外tf2的遍历dataset速度貌似比tf1 eager慢一些...
        if not gezi.get('tpu'):
          if distributed:
            gpus = [dist.local_rank()]
          physical_devices = tf.config.list_physical_devices('GPU')
          tf.config.set_visible_devices([physical_devices[x] for x in gpus],
                                        'GPU')
          tf.config.set_soft_device_placement(True)
          if FLAGS.allow_growth:
            try:
              for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
                # # 下面可以限制tf 不占用gpu空间，FIXME 但是目前有错误
                # # tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0 in order to run Equal: Dst tensor is not initialized. [Op:Equal]
                # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)])
            except Exception:
              pass
    else:
      if tf.__version__ < '2':
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = FLAGS.allow_growth
        config.allow_soft_placement = FLAGS.allow_soft_placement
        if distributed:
          config.gpu_options.visible_device_list = str(dist.local_rank())
        elif gpus:
          config.gpu_options.visible_device_list = ','.join(map(str, gpus))
        tf.compat.v1.enable_eager_execution(config=config)

        if FLAGS.tf_v2:
          tf.compat.v1.enable_v2_behavior()
      else:
        if not FLAGS.simple_startup:
          if distributed:
            gpus = [dist.local_rank()]
          physical_devices = tf.config.list_physical_devices('GPU')
          # print('--------', physical_devices, gpus)
          try:
            tf.config.set_visible_devices([physical_devices[x] for x in gpus],
                                          'GPU')
            tf.config.set_soft_device_placement(FLAGS.allow_soft_placement)
          except Exception:
            pass
          if FLAGS.allow_growth:
            try:
              for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(
                    gpu, FLAGS.allow_growth)
            except Exception:
              pass
  else:
    melt.get_session(allow_growth=FLAGS.allow_growth,
                     log_device_placement=FLAGS.log_device_placement,
                     gpus=gpus,
                     graph=graph)

  if not FLAGS.torch:
    if not FLAGS.use_tower_loss and not FLAGS.num_gpus:
      FLAGS.num_gpus = 1
      melt.set_global('num_gpus', 1)

  # # tf 1 with FLAGS.eager == 1 or tf 2 by default
  is_eager = tf.executing_eagerly()
  # assert FLAGS.eager == is_eager, f'{FLAGS.eager}, {is_eager}'
  # assert not (FLAGS.graph and is_eager)

  if not FLAGS.torch:
    if not is_eager:
      logging.info(
          f'Tf dataset and Tf model train in Graph mode, keras {FLAGS.keras}, distributed:{distributed}'
      )
    else:
      logging.info(
          f'Tf dataset and Tf model train in Eager mode, keras {FLAGS.keras}, distributed:{distributed}'
      )
  else:
    if not FLAGS.torch_only:
      logging.info(
          'Tf dataset and Torch model train in Eager mode, distributed:',
          distributed)
    else:
      logging.info('Torch dataset and Torch model train, distributed:',
                   distributed)

  if 'BIG' in os.environ and int(os.environ['BIG']) == 1:
    if FLAGS.big_batch_size is not None:
      FLAGS.batch_size = FLAGS.big_batch_size
    if FLAGS.big_buckets is not None:
      FLAGS.buckets = FLAGS.big_buckets
    if FLAGS.big_batch_sizes is not None:
      FLAGS.batch_sizes = FLAGS.big_batch_sizes

  if 'VLOG' in os.environ:
    FLAGS.log_level = int(os.environ['VLOG'])
  logging.info(
      'log_level:', FLAGS.log_level,
      '(try --debug to show more or --log_level=(> 20) to show less(no INFO), try --verbose to show train/valid loss intervaly)'
  )

  if 'NUM_EPOCHS' in os.environ:
    FLAGS.num_epochs = float(os.environ['NUM_EPOCHS'])
    logging.info('Using num_epochs set from env as %d' % FLAGS.num_epochs)

  if FLAGS.valid_interval_epochs > 0:
    FLAGS.valid_interval_epochs = min(FLAGS.valid_interval_epochs,
                                      FLAGS.num_epochs)
  elif FLAGS.valid_interval_epochs == 0:
    FLAGS.valid_interval_epochs = FLAGS.num_epochs
  logging.debug('valid_interval_epochs:', FLAGS.valid_interval_epochs)

  if FLAGS.test_interval_epochs > 0:
    FLAGS.test_interval_epochs = min(FLAGS.test_interval_epochs,
                                     FLAGS.num_epochs)
  elif FLAGS.test_interval_epochs == 0:
    FLAGS.test_interval_epochs = FLAGS.num_epochs
  logging.debug('test_interval_epochs:', FLAGS.test_interval_epochs)

  if 'NUM_STEPS' in os.environ:
    FLAGS.num_steps = int(os.environ['NUM_STEPS'])
    logging.info('Using num_steps set from env as %d' % FLAGS.num_steps)
    if FLAGS.num_steps < 0:
      FLAGS.train_only = True

  if 'EVAL_STEP' in os.environ:
    FLAGS.metric_eval_interval_steps = int(os.environ['EVAL_STEP'])
    FLAGS.valid_interval_steps = int(os.environ['EVAL_STEP'])

  if 'EVAL_STEPS' in os.environ:
    FLAGS.metric_eval_interval_steps = int(os.environ['EVAL_STEPS'])
    FLAGS.valid_interval_steps = int(os.environ['EVAL_STEPS'])

  if 'CVO' in os.environ:
    FLAGS.cv_valid_only = int(os.environ['CVO'])

  if 'CTO' in os.environ:
    FLAGS.cv_test_only = int(os.environ['CTO'])

  if 'VIE' in os.environ:
    FLAGS.valid_interval_epochs = float(os.environ['VIE'])

  if 'WRITE_VALID' in os.environ:
    FLAGS.write_valid = int(os.environ['WRITE_VALID'])

  if 'DO_TEST' in os.environ:
    FLAGS.do_test = int(os.environ['DO_TEST'])

  if FLAGS.mode == 'test':
    FLAGS.do_test = True

  if 'GPUS' in os.environ:
    FLAGS.gpus = int(os.environ['GPUS'])

  if not FLAGS.metric_eval:
    FLAGS.metric_eval_interval_steps = 0

  if 'LEARNING_RATE_DECAY_FACTOR' in os.environ:
    FLAGS.learning_rate_decay_factor = int(
        os.environ['LEARNING_RATE_DECAY_FACTOR'])
    logging.info('Using learning_rate_decay_factor set from env as %d' %
                 FLAGS.learning_rate_decay_factor)

  if 'BUCKETS' in os.environ:
    FLAGS.buckets = os.environ['BUCKETS']
    logging.info('Using buckets set from env as ', FLAGS.buckets)

  if 'BATCH_SIZES' in os.environ:
    FLAGS.batch_sizes = os.environ['BATCH_SIZES']
    logging.info('Using batch sizes set from env as ', FLAGS.batch_sizes)

  if 'NUM_LAYERS' in os.environ:
    FLAGS.num_layers = int(os.environ['NUM_LAYERS'])
    logging.info('Using num layers set from env as ', FLAGS.num_layers)

  if 'TRAIN_INPUT' in os.environ:
    FLAGS.train_input = os.environ['TRAIN_INPUT']

  if 'VALID_INPUT' in os.environ:
    FLAGS.valid_input = os.environ['VALID_INPUT']

  # TEST means also inference
  if 'TEST_INPUT' in os.environ:
    if os.environ['TEST_INPUT'] == '1':
      if not FLAGS.test_input:
        assert FLAGS.train_input
        assert 'train' in FLAGS.train_input
        FLAGS.test_input = FLAGS.train_input.replace('train',
                                                     'test').split(',')[0]
    else:
      FLAGS.test_input = os.environ['TEST_INPUT']
  # else:
  #   if not ('TRAIN_ALL' in os.environ and int(os.environ['TRAIN_ALL']) != 0):
  #     FLAGS.test_input = None

  if FLAGS.test_input == '1':
    assert FLAGS.train_input
    assert 'train' in FLAGS.train_input
    FLAGS.test_input = FLAGS.train_input.replace('train', 'test').split(',')[0]
  elif FLAGS.test_input == '0':
    FLAGS.test_input = None

  if FLAGS.valid_input == '1':
    assert FLAGS.train_input
    assert 'train' in FLAGS.train_input
    FLAGS.valid_input = FLAGS.train_input.replace('train',
                                                  'valid').split(',')[0]
  elif FLAGS.valid_input == '0':
    FLAGS.valid_input = None

  assert FLAGS.dataset_rate >= 1, 'not support like 0.4 * dataset now'

  if FLAGS.valid_input:
    FLAGS.fold = None

  if FLAGS.fold is not None:
    FLAGS.num_folds = None

  if 'TRAIN_ALL' in os.environ and int(os.environ['TRAIN_ALL']) != 0:
    FLAGS.train_all = True
    # FLAGS.buckets = None  # no buckets for train all mode TODO might still need buket

    if FLAGS.fold is not None:
      if FLAGS.model_dir:
        FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'all')

    # also evluate on fold 0 if not set fold
    if FLAGS.fold is None:
      FLAGS.fold = 0

    if not FLAGS.test_input:
      assert FLAGS.train_input
      assert 'train' in FLAGS.train_input
      FLAGS.test_input = FLAGS.train_input.replace('train',
                                                   'test').split(',')[0]

  if 'TRAIN_ONLY' in os.environ and int(os.environ['TRAIN_ONLY']) != 0:
    FLAGS.train_all = True
    FLAGS.train_only = True

    if FLAGS.fold is not None:
      if FLAGS.model_dir:
        FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'all')

  if FLAGS.train_only:
    FLAGS.test_input = None
    FLAGS.valid_input = None
    FLAGS.metric_eval = False
    FLAGS.async_valid = False

  if 'WRITE_VALID' in os.environ:
    FLAGS.write_valid = bool(os.environ['WRITE_VALID'])

  if 'MIN_AFTER_DEQUEUE' in os.environ:
    FLAGS.min_after_dequeue = int(os.environ['MIN_AFTER_DEQUEUE'])

  if 'BUFFER_SIZE' in os.environ:
    FLAGS.buffer_size = int(os.environ['BUFFER_SIZE'])

  if 'RANDOM_EMB' in os.environ and os.environ['RANDOM_EMB'] == '1':
    FLAGS.word_embedding_file = None

  if not FLAGS.distributed:
    melt.set_global('batch_size',
                    FLAGS.batch_size * max(melt.num_gpus(), dist.size()))
    melt.set_global('eval_batch_size',
                    FLAGS.eval_batch_size * max(melt.num_gpus(), dist.size()))
  else:
    melt.set_global('batch_size', FLAGS.batch_size * dist.size())
    if FLAGS.horovod_eval:
      melt.set_global('eval_batch_size', FLAGS.eval_batch_size * dist.size())
    else:
      melt.set_global('eval_batch_size', FLAGS.eval_batch_size)

  melt.set_global('num_gpus2', max(int(melt.batch_size() / FLAGS.batch_size),
                                   1))

  num_gpus_ = int(melt.batch_size() /
                  FLAGS.batch_size) if not gezi.is_cpu_only() else 0
  if tpu:
    num_gpus_ = FLAGS.num_gpus
  gezi.set('num_gpus', num_gpus_)

  if FLAGS.learning_rate_scale_bygpu:
    FLAGS.learning_rate *= (melt.batch_size() / ori_batch_size)
    if FLAGS.learning_rates:
      FLAGS.learning_rates = [
          lr * (melt.batch_size() / ori_batch_size)
          for lr in FLAGS.learning_rates
      ]
  if FLAGS.learning_rate_multiplier is not None:
    FLAGS.learning_rate *= FLAGS.learning_rate_multiplier
    if FLAGS.learning_rates:
      FLAGS.learning_rates = [
          lr * FLAGS.learning_rate_multiplier for lr in FLAGS.learning_rates
      ]
  
  global_batch_size = melt.batch_size()
  acc_steps = FLAGS.gradient_accumulation_steps
  replica_batch_size = FLAGS.batch_size
  eval_batch_size = melt.eval_batch_size()
  num_gpus = num_gpus_
  gpu = gpus_global
  CUDA_VISIABLE_DEVICES = gezi.get_specific_gpus()
  work_mode = FLAGS.work_mode
  seed = FLAGS.seed
  # notice melt.batch_size() is actual batch size, FLAGS.batch_size is actually batch size per gpu!
  logging.debug('global_batch_size:', melt.batch_size(), 'acc_steps:', FLAGS.gradient_accumulation_steps, 
               'replica_batch_size:', FLAGS.batch_size, 'eval_batch_size:', melt.eval_batch_size(),
               'num_gpus:', num_gpus_, 'gpu:', gpus_global,
               f'CUDA_VISIABLE_DEVICES={gezi.get_specific_gpus()}',
               'work_mode:', FLAGS.work_mode, 'distributed:', distributed, 'seed:', FLAGS.seed
              #  'horovod:', FLAGS.use_horovod
               )
  ic(global_batch_size, acc_steps, replica_batch_size, eval_batch_size, num_gpus, gpus, CUDA_VISIABLE_DEVICES, work_mode, seed)

  if tpu is None:
    _set_strategy()

  if FLAGS.learning_rate_patience:
    FLAGS.dynamic_learning_rate = True
    if not FLAGS.learning_rate_decay_factor:
      FLAGS.learning_rate_decay_factor = 0.5

  # TODO learning_rate_weight is deprecated But if now remove will not run eager tf ok
  # (`Checkpoint` was expecting a trackable object (an object derived from `TrackableBase`), got <melt.util.LearningRate object at 0x7f96485781d0>.)
  try:
    if not is_eager and not FLAGS.torch_only:
      learning_rate_weight = tf.compat.v1.get_variable('learning_rate_weight',
                                                       initializer=tf.ones(
                                                           shape=(),
                                                           dtype=tf.float32),
                                                       trainable=False)
      tf.compat.v1.add_to_collection('learning_rate_weight',
                                     learning_rate_weight)
      if FLAGS.num_learning_rate_weights > 0:
        learning_rate_weights = tf.compat.v1.get_variable(
            'learning_rate_weights',
            initializer=tf.ones(shape=(FLAGS.num_learning_rate_weights),
                                dtype=tf.float32),
            trainable=False)
        tf.compat.v1.add_to_collection('learning_rate_weights',
                                       learning_rate_weights)
    else:
      # will cause torch each process occupy all gpus...   TODO tf torch conflict!
      if is_eager:
        # learning_rate_weight = tfe.Variable(1., name='learning_rate_weight', trainable=False)
        learning_rate_weight = tf.Variable(1.,
                                           name='learning_rate_weight',
                                           trainable=False)
        # tf.add_to_collection('learning_rate_weight', learning_rate_weight)
        melt.set_global('learning_rate_weight', learning_rate_weight)
        if FLAGS.num_learning_rate_weights > 0:
          # learning_rate_weights = tfe.Variable(tf.ones(shape=(FLAGS.num_learning_rate_weights), dtype=tf.float32), name='learning_rate_weights', trainable=False)
          learning_rate_weights = tf.Variable(tf.ones(
              shape=(FLAGS.num_learning_rate_weights), dtype=tf.float32),
                                              name='learning_rate_weights',
                                              trainable=False)
          # tf.add_to_collection('learning_rate_weights', learning_rate_weights)
          melt.set_global('learning_rate_weights', learning_rate_weights)
      else:
        melt.set_global('learning_rate_weight', melt.LearningRate(1.))
    # melt.set_global('learning_rate_weight', melt.LearningRate(1.))
  except Exception:
    pass

  if FLAGS.train_loop:
    FLAGS.start_epoch = 0

  if FLAGS.valid_span < 0:
    FLAGS.loop_fixed_valid = True

  if not FLAGS.train_hour:
    FLAGS.train_hour = FLAGS.valid_hour

  if FLAGS.metric_eval and FLAGS.valid_hour:
    os.system(f'mkdir -p {FLAGS.log_dir}/infos/{FLAGS.valid_hour}')

  # if not FLAGS.use_horovod or hvd.rank() == 0:
  if FLAGS.train_loop:
    root = FLAGS.train_input.split(',')[0].strip()
    dirs = glob.glob('%s/*' % root)
    assert dirs, root
    # dirs = [x for x in dirs if os.path.isdir(x) and len(os.path.basename(x)) == len('2019121612') and os.path.basename(x).isdigit()]
    dirs = [
        x for x in dirs if os.path.isdir(x) and os.path.basename(x).isdigit()
    ]
    # dirs = [x for x in dirs if gezi.getdirsize(x, 'g') > 10]
    if not FLAGS.loop_range:
      dirs = sorted(dirs, key=lambda x: os.path.basename(x))
    else:
      dirs = sorted(dirs, key=lambda x: int(os.path.basename(x)))
    start_hour = FLAGS.start_hour
    end_hour = FLAGS.end_hour

    latest_model = None
    start_hour_from_model = None
    if FLAGS.model_dir:
      latest_model = melt.latest_checkpoint(FLAGS.model_dir)
    logging.debug('latest_model:', latest_model)

    if latest_model:
      try:
        hour_str = os.path.basename(latest_model).split('-')[1]
        # d is the last train hour
        if not FLAGS.loop_range:
          if len(hour_str) == len('2020050100'):
            d = datetime.strptime(hour_str, '%Y%m%d%H')
            # so from model path we can infer now start hour is d + 1
            start_hour_from_model = (d +
                                     timedelta(hours=1)).strftime('%Y%m%d%H')
          else:
            d = datetime.strptime(hour_str, '%Y%m%d')
            start_hour_from_model = (d + timedelta(days=1)).strftime('%Y%m%d')
        else:
          start_hour_from_model = str(int(hour_str) + 1)
      except Exception:
        logging.info('no start_hour get from model', latest_model)

    if not start_hour:
      start_hour = start_hour_from_model
    else:
      logging.info(
          f'input start_hour [{start_hour}] is different from model last saved hour + 1 as [{start_hour_from_model}]'
      )
      if start_hour_from_model and start_hour < start_hour_from_model and not FLAGS.force_start:
        logging.info(
            'start hour before model last train hour will ignore user input start_hour, still use start_hour_from_model'
        )
        start_hour = start_hour_from_model
    if FLAGS.work_mode != 'train':
      if not FLAGS.valid_hour:
        if not FLAGS.loop_range:
          FLAGS.valid_hour = (
              datetime.strptime(start_hour, '%Y%m%d%H') +
              timedelta(hours=FLAGS.valid_span)).strftime('%Y%m%d%H')
        else:
          FLAGS.valid_hour = str(int(start_hour) + FLAGS.valid_span)
        FLAGS.valid_span = 0

    if FLAGS.loop_latest:
      if not start_hour:
        if not end_hour:
          end_hour = os.path.basename(dirs[-1])
        if FLAGS.hours:
          start_hour = (datetime.strptime(end_hour, '%Y%m%d%H') +
                        timedelta(hours=-FLAGS.hours)).strftime('%Y%m%d%H')
        elif FLAGS.days:
          start_hour = (datetime.strptime(end_hour, '%Y%m%d%H') +
                        timedelta(-FLAGS.days)).strftime('%Y%m%d%H')
      elif not end_hour:
        if FLAGS.hours:
          end_hour = (datetime.strptime(start_hour, '%Y%m%d%H') +
                      timedelta(hours=FLAGS.hours)).strftime('%Y%m%d%H')
        elif FLAGS.days:
          end_hour = (datetime.strptime(start_hour, '%Y%m%d%H') +
                      timedelta(FLAGS.days)).strftime('%Y%m%d%H')
    else:
      if not end_hour:
        if not start_hour:
          start_hour = os.path.basename(dirs[0])
        if FLAGS.hours:
          end_hour = (datetime.strptime(start_hour, '%Y%m%d%H') +
                      timedelta(hours=FLAGS.hours)).strftime('%Y%m%d%H')
        elif FLAGS.days:
          end_hour = (datetime.strptime(start_hour, '%Y%m%d%H') +
                      timedelta(FLAGS.days)).strftime('%Y%m%d%H')
      elif not start_hour:
        if FLAGS.hours:
          start_hour = (datetime.strptime(end_hour, '%Y%m%d%H') +
                        timedelta(hours=-FLAGS.hours)).strftime('%Y%m%d%H')
        elif FLAGS.days:
          start_hour = (datetime.strptime(end_hour, '%Y%m%d%H') +
                        timedelta(-FLAGS.days)).strftime('%Y%m%d%H')

    if not end_hour:
      end_hour = os.path.basename(dirs[-1])

    if not start_hour:
      logging.warning(
          'You may need to set FLAGS.start_hour or FLAGS.hours or FLAGS.days, now will start from oldest record'
      )
      start_hour = os.path.basename(dirs[0])

    if start_hour == end_hour and not FLAGS.train_only and FLAGS.work_mode == 'train' and not FLAGS.loop_train_all:
      raise ValueError(f'start hour == end hour do nothing {start_hour}')

    logging.info('start_hour', start_hour, 'end_hour', end_hour)

    if FLAGS.work_mode == 'train':
      if not FLAGS.loop_range:
        dirs = [
            x for x in dirs if os.path.basename(x) >= start_hour and
            os.path.basename(x) <= end_hour
        ]
      else:
        dirs = [
            x for x in dirs if int(os.path.basename(x)) >= int(start_hour) and
            int(os.path.basename(x)) <= int(end_hour)
        ]

      if FLAGS.rounds and not FLAGS.loop_train_all:
        # HACK to make get_num_records_from_dirs not cost too much
        if FLAGS.loop_type == 'hour':
          limit = (FLAGS.rounds + 2) * 2
        else:
          limit = (FLAGS.rounds + 2) * 24
        dirs = dirs[:limit]
      num_ori_dirs = len(dirs)
      dirs = [
          x for x in tqdm(dirs, ascii=True, desc='get_num_records_from_dirs') if
          melt.get_num_records_from_dir(x, recount=True) >= FLAGS.min_tfrecords
      ]

      logging.info('Num dirs for train loop', len(dirs), 'out of', num_ori_dirs)

      if FLAGS.loop_type == 'hour':
        FLAGS.train_input = '|'.join(dirs)
        # FLAGS.valid_interval_epochs = 1.
      elif FLAGS.loop_type == 'day':
        # FLAGS.valid_span = 1
        FLAGS.eval_days = False
        FLAGS.save_interval_epochs = FLAGS.valid_interval_epochs
        days = gezi.chunks(dirs, 24)
        days = [','.join(x) for x in days]
        assert len(
            days
        ) > 1, 'Nont enough hours data for daily training, start_hour %s, end_hour %s, model %s' % (
            start_hour, end_hour, latest_model)
        FLAGS.train_input = '|'.join(days)
      else:
        raise ValueError(FLAGS.train_loop)
    else:
      assert FLAGS.valid_hour

      count = 0
      while True:
        dirs_ = [x for x in dirs if os.path.basename(x) == FLAGS.valid_hour]
        if not dirs_ or melt.get_num_records_from_dir(
            dirs_[0], recount=True) < FLAGS.min_tfrecords:
          dirs_ = []
          FLAGS.valid_hour = (datetime.strptime(FLAGS.valid_hour, '%Y%m%d%H') +
                              timedelta(hours=1)).strftime('%Y%m%d%H')
          logging.info(count, FLAGS.valid_hour)
          count += 1
          if count == 10:
            logging.error('could not find valid hour tfrecord')
            exit(0)
        if dirs_:
          dirs = dirs_
          break
      FLAGS.train_input = '|'.join(dirs)

  if FLAGS.valid_hour and not FLAGS.version:
    FLAGS.version = FLAGS.valid_hour

  if FLAGS.async_valid:
    if FLAGS.valid_interval_epochs and FLAGS.valid_interval_epochs <= 1. and FLAGS.valid_interval_epochs > 0:
      if not FLAGS.save_interval_epochs and FLAGS.save_interval_epochs < FLAGS.valid_interval_epochs and \
          int(FLAGS.valid_interval_epochs / FLAGS.save_interval_epochs) == (FLAGS.valid_interval_epochs / FLAGS.save_interval_epochs):
        FLAGS.save_interval_epochs = FLAGS.valid_interval_epochs

  if FLAGS.model_dir != FLAGS.log_dir:
    logging.info(f'model: [{FLAGS.model_name}]', 'model_dir:',
                 f'[{FLAGS.model_dir}]', 'log_dir:', FLAGS.log_dir)
  else:
    logging.info(f'model: [{FLAGS.model_name}]', 'model_dir:',
                 f'[{FLAGS.model_dir}]')

  if FLAGS.restore_include2:
    FLAGS.restore_include = FLAGS.restore_include2 if not FLAGS.restore_include else FLAGS.restore_include + \
        ',' + FLAGS.restore_include2
  if FLAGS.restore_exclude2:
    FLAGS.restore_exclude = FLAGS.restore_exclude2 if not FLAGS.restore_exclude else FLAGS.restore_exclude + \
        ',' + FLAGS.restore_exclude2

  if FLAGS.work_mode == 'valid':
    FLAGS.restore_exclude = ''
    FLAGS.restore_include = ''

  # TODO xla not work
  # if FLAGS.enable_xla:
  #     if FLAGS.hack_device:
  #         FLAGS_hack_device = 'xla_' + FLAGS.hack_device
  FLAGS.global_batch_size = melt.batch_size()
  FLAGS.replica_batch_size = FLAGS.batch_size
  FLAGS.global_eval_batch_size = melt.eval_batch_size()
  FLAGS.replica_eval_batch_size = FLAGS.eval_batch_size
  melt.set_global('replica_batch_size', FLAGS.batch_size)
  melt.set_global('replica_eval_batch_size', FLAGS.eval_batch_size)
  strategy = melt.get_strategy()
  FLAGS.num_replicas = strategy.num_replicas_in_sync

  if FLAGS.fp16:
    melt.set_fp16_policy()
  else:
    gezi.set('precision_policy_name', 'float32')

  logging.info('Train precision:', gezi.get('precision_policy_name'))

  if FLAGS.enable_xla:
    tf.config.optimizer.set_jit(True)

  if FLAGS.update_each_epoch is None:
    if gezi.in_notebook():
      FLAGS.update_each_epoch = False
    else:
      FLAGS.update_each_epoch = True

  if FLAGS.reset_all:
    FLAGS.reset_global_step = True
    FLAGS.reset_learning_rate = True

  if FLAGS.reset_global_step:
    logging.info(
        f'reset_global_step to 0, remove model_step.txt total_step.txt from {FLAGS.model_dir}'
    )
    gezi.try_remove(f'{FLAGS.model_dir}/model_step.txt')
    gezi.try_remove(f'{FLAGS.model_dir}/total_step.txt')

  # ------------wandb
  if not gezi.get('wandb_run'):
    try:
      if FLAGS.wandb_silent is None:
        # FLAGS.wandb_silent = False
        if gezi.in_colab():
          FLAGS.wandb_silent = False
        else:
          FLAGS.wandb_silent = True
      if FLAGS.wandb_silent:
        os.environ["WANDB_SILENT"] = "true"
      import wandb
      if FLAGS.wandb:
        wandb.login(key=FLAGS.wandb_key)
        wandb_config = gezi.get('wandb_config', FLAGS.flag_values_dict())
        wandb_id = gezi.read_str_from(f'{FLAGS.log_dir}/wandb_id.txt')
        if wandb_id:
          FLAGS.wandb_resume = True
          FLAGS.wandb_id = wandb_id
        run = wandb.init(project=FLAGS.wandb_project,
                         group=FLAGS.wandb_group or
                         os.path.basename(os.path.dirname(FLAGS.model_dir)),
                         dir=FLAGS.wandb_dir or
                         os.path.dirname(FLAGS.model_dir),
                         config=wandb_config,
                         name=FLAGS.wandb_name or FLAGS.model_name,
                         notes=FLAGS.wandb_notes,
                         tags=FLAGS.wandb_tags,
                         id=FLAGS.wandb_id,
                         resume=FLAGS.wandb_resume,
                         sync_tensorboard=FLAGS.wandb_tb,
                         magic=FLAGS.wandb_magic)
        logging.info('wand_url:', run.url)
        gezi.set('wandb_run', run)
        wandb_id = wandb.run.id
        gezi.set('wandb_id', wandb_id)
        FLAGS.wandb_id = wandb_id
        gezi.write_to_txt(wandb_id, f'{FLAGS.log_dir}/wandb_id.txt')
        if FLAGS.gcs_sync and FLAGS.gcs_root:
          if not FLAGS.gcs_dest:
            FLAGS.gcs_dest = f'{FLAGS.gcs_root}/{FLAGS.wandb_id}'
    except Exception as e:
      logging.warning(e)

  # TODO wandb 当前 不支持异步
  if FLAGS.wandb:
    FLAGS.async_eval = False

  if FLAGS.gcs_dest:
    logging.info('gcs_dest:', FLAGS.gcs_dest)

  # # HACKS
  # if (tf.__version__ >= '2.4' and
  #     tf.__version__ < '2.6') and (FLAGS.num_gpus and FLAGS.num_gpus != 1):
  #   FLAGS.valid_interval_steps = 0  # 否则总是展示warning 屏蔽掉就不要记录val loss了

  # --------save init run info to txt
  history_flag_file = os.path.join(FLAGS.log_dir, 'history_flags.txt')
  flag_file = os.path.join(FLAGS.log_dir, 'flags.txt')
  command_file = os.path.join(FLAGS.log_dir, 'command.txt')
  args_file = os.path.join(FLAGS.log_dir, 'args.txt')
  script_file = os.path.join(FLAGS.log_dir, 'script.txt')
  try:
    scripts = psutil.Process().parent().cmdline()
  except Exception:
    scripts = ''
  if FLAGS.mode != 'async_valid':
    try:
      os.remove(flag_file)
    except Exception:
      pass
    FLAGS.append_flags_into_file(history_flag_file)
    FLAGS.append_flags_into_file(flag_file)
    # FLAGS.append_flags_into_file(os.path.join(FLAGS.log_dir, 'log.html'))
    gezi.write_to_txt(FLAGS.command, command_file)
    gezi.write_to_txt(' '.join([f'"{x}"' for x in sys.argv[1:]]), args_file)
    gezi.write_to_txt(' '.join(scripts), script_file)

  with open(f"{FLAGS.log_dir}/flags.json", 'w') as fout:
    json_dumps_str = json.dumps(FLAGS.flag_values_dict(), indent=4)
    print(json_dumps_str, file=fout)

  try:
    # tf 2.2 AttributeError: Can't pickle local object 'DEFINE_alias.<locals>._Parser'
    with open(f'{FLAGS.log_dir}/flags.pkl', 'wb') as fout:
      pickle.dump(flag_values, fout)
  except Exception:
    pass

  with open(f"{FLAGS.log_dir}/global.json", 'w') as fout:
    json_dumps_str = gezi.safe_serialize(gezi.global_dict)
    print(json_dumps_str, file=fout)

  gezi.set('FLAGS', FLAGS.flag_values_dict())

  # TODO might need to un lock after one training step 但是tf graph模式init之后到第一个step完成准备时间很长 会造成其它进程较长时间等待
  # 当前存在冲突可能性因为tf程序到这里可能只占用了270M显存，但是程序循环计算起来占用超过8g
  # 而一个torch程序可能需要10g显存 到这里看到显存够用也占用了这个显存后续就会冲突OOM，当然可也设置torch 也有min used gpu 限制20m
  # 不过对于torch验证程序由于无法释放显存长时间占用比较浪费 其实完全可以一个显卡同时跑两个torch验证程序。
  # tf目前采用了限制20m 也就是限制只能一个做程序做验证 但是tf能中途释放显存占用时间较短 infer完成之后的valuate不占用显存。
  if fp:
    fcntl.flock(fp, fcntl.LOCK_UN)
    fp.close()
    fp = None

  gezi.set('gpus', gpus)
  gezi.set('inited', True)
  logging.debug('-------------done melt.init using:', timer.elapsed())
