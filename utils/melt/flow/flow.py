#!/usr/bin/env python
# ==============================================================================
#          \file   flow.py
#        \author   chenghuige  
#          \date   2016-08-17 10:48:46.141744
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

if tf.__version__ < '2':
  ## TODO for tf2 not ok even with tf_slim
  #import tensorflow.contrib.slim as slim
  import tf_slim as slim

try:
  from tensorflow.contrib import tpu
  from tensorflow.python.training.monitored_session import Scaffold
except Exception:
  pass 


import os, sys, traceback
import melt 
import gezi
logging = gezi.logging
from gezi.summary import SummaryWriter
import glob
import inspect
import traceback
import time
import numpy as np
import fcntl
import pandas as pd
import subprocess

from gezi import tqdm

#sync = 'rsync -a --update'
sync = 'scp -r'

def _import_horovod():
  if not FLAGS.torch:
    import horovod.tensorflow as hvd
  else:
    import torch
    import horovod.torch as hvd
  return hvd

def init_iter(sess, iter, subset, index=0):
  if iter is not None and hasattr(iter, 'initializer'):
    need_feed = FLAGS.train_loop and FLAGS.rounds > 1 and not tf.executing_eagerly() and FLAGS.feed_dataset
    if not need_feed:
      sess.run(iter.initializer)
    else:
      feed_name = f'{subset}_{index}'
      sess.run(iter.initializer, feed_dict={gezi.get_global(feed_name): melt.Dataset.get_filenames_(subset, shuffle=FLAGS.shuffle)})

def init_iters(sess, index=0):
  iter = gezi.get_global('iter')
  init_iter(sess, iter, 'train')
  valid_iter = gezi.get_global('valid_iter', None)
  init_iter(sess, valid_iter, 'valid')
  valid_iter2 = gezi.get_global('valid_iter2', None)
  init_iter(sess, valid_iter2, 'valid', 1)
  test_iter = gezi.get_global('test_iter', None)
  init_iter(sess, test_iter, 'test')
  # print(sess, FLAGS.task_index)

def tf_flow(process_once, model_dir=None, num_steps=None, sess=None):
  """
  basic flow for tf records, allow most freedom for usage, if not tfrecords no need for flow
  Args:
  train_once: function with 2 inputs sess and step
  """
  init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
  if sess is None:
    sess = tf.compat.v1.InteractiveSession()

  if not model_dir:
    sess.run(init_op)
  else:
    melt.restore(sess, model_dir)

  try:
    step = 0
    #while not coord.should_stop():
    while True:
      stop = process_once(sess, step)
      if stop is True:
        print('Early stop running %d stpes'%(step))
        raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
      step += 1
      if num_steps and step == num_steps:
        raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  except tf.errors.OutOfRangeError:
    print('Done training for %d steps.' % (step))
  # finally:
  #   coord.request_stop()
  # coord.join(threads)

  sess.close()
  return step

def dist_flow(process_once):
  server = gezi.get('server')
  # with tf.train.MonitoredTrainingSession(master=server.target,
  #                                        is_chief=(FLAGS.task_index == 0),
  #                                        checkpoint_dir=FLAGS.model_dir) as sess:
  hooks = [tf.train.StopAtStepHook(last_step=1500)]
  sess  = tf.train.MonitoredTrainingSession(master=server.target,
                                         is_chief=(FLAGS.task_index == 0),
                                         checkpoint_dir=FLAGS.model_dir,
                                         hooks=hooks)
  sess.graph._unsafe_unfinalize()
  init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                      tf.compat.v1.local_variables_initializer())
  sess.run(init_op)
  init_iters(sess)
  global_step = tf.train.get_or_create_global_step()
  step = 0
  with gezi.Timer('Dist run', print_fn=print) as timer:
    while not sess.should_stop():
      step += 1
      process_once(sess, step)
      gstep = sess.run(global_step)
      if gstep % 10 == 0:
        print(step, gstep, FLAGS.task_index)
  print('dist run:', step, gstep, FLAGS.task_index)
 
def _get_model_path(model_dir, save_model=False):
  if os.path.exists(model_dir + '.index') and os.path.exists(model_dir + '.meta'):
    # input is real model path
    return model_dir
  if not os.path.exists(model_dir):
    if save_model:
      gezi.try_mkdir(model_dir)
    return None
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #input valid dir and return latest model
    return os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  elif os.path.isdir(model_dir):
    #input valid dir but no models
    return None 
  else:
    #this might be user specified model like ./model/model-100.ckpt
    #the file exists and we NOTICE we do not check if it is valid model file!
    return model_dir

def _get_checkpoint_path(checkpoint_path, step=None, num_steps_per_epoch=None, epoch=None):
  if not num_steps_per_epoch:
    return checkpoint_path
  if not FLAGS.train_hour:
    return '%s-%.2f'%(checkpoint_path, epoch or step / float(num_steps_per_epoch))
  else:
    return '%s-%s-%.2f'%(checkpoint_path, FLAGS.train_hour, epoch or step / float(num_steps_per_epoch))

def _async_valid():
  if not FLAGS.metric_eval:
    return

  if not FLAGS.do_valid:
    return

  if FLAGS.local_rank != 0:
    return 

  del_model_path = FLAGS.del_model_path
  latest_checkpoint = melt.latest_checkpoint(FLAGS.model_dir)

  command = FLAGS.command

  if FLAGS.masked_fields:
    command = command.replace('|', '+++')

  # --max_used_gpu_mem=20 removed
  command += ' --train_only=0 --mode=async_valid --write_summary --train_loop=0 --loop_train=0 --valid_interval_epochs=1 --clear_model=0  save_model=0 --async_valid=0 --rounds=1 --ps_strategy=0'
  if FLAGS.train_loop:
    command += ' --valid_input=' + FLAGS.valid_input + ' --train_input=' + FLAGS.train_input \
                + ' --valid_hour=' + FLAGS.valid_hour + ' --eval_round=%d' % FLAGS.eval_round \
                + ' --from_loop_train'
  if latest_checkpoint:
    command += f' --model_path={latest_checkpoint}'
  if del_model_path:
    command += f' --del_model_path={del_model_path}'
  if FLAGS.total_time:
    command += f' --total_time={FLAGS.total_time}'
  if FLAGS.train_time:
    command += f' --train_time={FLAGS.train_time}'
  if FLAGS.valid_time:
    command += f' --valid_time={FLAGS.valid_time}'
  if FLAGS.l2_:
    command += f' --l2_={FLAGS.l2_}'
  if FLAGS.params_:
    command += f' --params_={FLAGS.params_}'
  if FLAGS.masked_fields:
    command += f" --masked_fields='{FLAGS.masked_fields}'"
    
  if FLAGS.model_name:
    command += f" --mn={FLAGS.model_name}"
    
  if FLAGS.is_last_eval is not None:
    command += f" --is_last_eval={int(FLAGS.is_last_eval)}"

  env = os.environ.copy()
  if 'CUDA_VISIBLE_DEVICES' in env and env['CUDA_VISIBLE_DEVICES'] != '-1':
    del env['CUDA_VISIBLE_DEVICES']
  if FLAGS.cpu_valid:
    env['CUDA_VISIBLE_DEVICES'] = '-1'
  elif FLAGS.num_valid_gpus > 1:
    command = 'horovodrun -np %d %s' % (FLAGS.num_valid_gpus, command)
  elif FLAGS.torch:
    #NOTICE tf could not do this, will cuda fail,select gpu in melt.init
  # else:
    # if torch using eager read tfrecord currently set device not work so need to force using 1 gpu
    gpus = gezi.get_gpus(FLAGS.min_free_gpu_mem, FLAGS.max_used_gpu_mem, env_limit=False)
    gpu = gpus[0] if gpus else -1
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['RANK'] = '0'
    env['LOCAL_RANK'] = '0'
  if not command.endswith('&'):
    command += ' &'
  # gezi.system(command) 
  env['WORLD_SIZE'] = '1'

  subprocess.Popen(command.split(), env=env)

def _write_metric_hours_file(model_dir, gstep, is_base=False):
  if not is_base:
    metric_hour_file = os.path.join(model_dir, 'metrics.csv')
    res = gezi.get('result')
  else:
    metric_hour_file = os.path.join(model_dir, 'base_metrics.csv')
    res = gezi.get('online_result')

  if not res:
    return

  name = 'hour' if FLAGS.valid_hour else 'step'
  
  try:
    df = pd.read_csv(metric_hour_file)
  except Exception:
    df = pd.DataFrame()

  try:
    value = FLAGS.valid_hour if FLAGS.valid_hour else str(gstep)
    res[name] = value
    names2 = [name, 'timestamp']
    res['timestamp'] = int(time.time())
    if FLAGS.train_hour:
      res['train_hour'] = FLAGS.train_hour
      names2 += ['train_hour']
    
    df = df.append(res, ignore_index=True)
    df.to_csv(metric_hour_file, index=False, float_format='%.4f')
  except Exception:
    logging.error(traceback.format_exc(), is_base)
    pass

def _write_metric_file(model_dir, gstep, is_base=False):
  if not is_base:
    res = gezi.get('result')
    metric_file = 'metrics.csv'
  else:
    res = gezi.get('online_result')
    metric_file = 'base_metrics.csv'

  value = FLAGS.valid_hour if FLAGS.valid_hour else str(gstep)
  root = f'{model_dir}/infos/{value}'
  os.system(f'mkdir -p {root}')
  
  if not res:
    os.system(f'touch {root}/no_metrics.txt')
    return

  name = 'hour' if FLAGS.valid_hour else 'step'
  
  try:
    metric_file = f'{root}/{metric_file}'
    res[name] = value
    names2 = [name, 'timestamp']
    res['timestamp'] = int(time.time())
    if FLAGS.train_hour:
      res['train_hour'] = FLAGS.train_hour
      names2 += ['train_hour']
    df = pd.DataFrame([res])
    df.to_csv(metric_file, index=False, float_format='%.4f')
  except Exception:
    logging.error(traceback.format_exc(), is_base)
    pass

def _try_eval_day():
  if FLAGS.metric_eval and FLAGS.eval_days:
    model_root = os.path.dirname(FLAGS.log_dir)
    model_name = os.path.basename(FLAGS.log_dir)
    command = f'CUDA_VISIBLE_DEVICES=-1 ./tools/eval-days.py {model_root} --models={model_name} --eval_step=1 --tfrecord_base={int(FLAGS.eval_days_online)} --group_by_impression={int(FLAGS.eval_group_by_impression)} &'
    gezi.system(command)

def _try_eval(model_dir, log_dir, metric_eval_fn, inference_fn=None):
  if not metric_eval_fn:
    return 
  if not FLAGS.do_valid:
    return
  use_horovod = FLAGS.use_horovod
  if use_horovod:
    hvd = _import_horovod()
  rank = FLAGS.local_rank
  if FLAGS.async_valid and FLAGS.ev_first and rank == 0:
    _async_valid()
  elif FLAGS.work_mode != 'train' or gezi.get_env('EVFIRST') == '1' or FLAGS.ev_first or gezi.get_env('INFER') == '1':
    assert os.path.isdir(FLAGS.model_dir), FLAGS.model_dir  
    if 'test' in FLAGS.work_mode or gezi.get_env('TEST') == '1' or gezi.get_env('INFER') == '1':
      inference_fn(FLAGS.model_dir)
      exit(0)
    else:
      step = melt.get_eval_step()
      logging.debug(FLAGS.valid_hour, step, melt.get_eval_step(from_file=True))
      model_path_ = _get_model_path(model_dir)
      if FLAGS.model_path:
        model_path_ = FLAGS.model_path
      
      names, vals = metric_eval_fn(model_path_)
      names, vals = list(names), list(vals)
      names_ = [x.replace('eval/', '') for x in names]
      gezi.set('metric_names', list(names_))
      gezi.set('metric_values', list(vals)) 
      gezi.set('result', dict(zip(names_, vals)))
      if len(names) > 15:
        if FLAGS.version in ['train', 'valid', 'test']:
          logging.info('valid_metrics:{}'.format(['%s:%.4f' % (name, val) for name, val in zip(names_[:30], vals[:30]) if not isinstance(val, str)] + ['version:{}'.format(FLAGS.version)]))
        else:
          logging.info('valid_metrics:{}'.format(['%s:%.4f' % (name, val) for name, val in zip(names_[:30], vals[:30]) if not isinstance(val, str)]))
      else:
        results = dict(zip(names_, vals))
        gezi.pprint_dict(gezi.dict_rename(results, 'Metrics/', ''), print_fn=logging.info, 
                        desc=f'train:[{FLAGS.train_hour}] valid:[{FLAGS.valid_hour}] valid_metrics:',
                        format='%.4f')

      if rank == 0:
        # TODO write key:value format each line, otherwise difficult to use if you add more metrics
        if FLAGS.valid_input and FLAGS.metric_values or gezi.get('metric_values'):
          _write_metric_file(log_dir, step)
          _write_metric_file(log_dir, step, is_base=True)
          
          with gezi.FileLock(f'{FLAGS.log_dir}/metric.lock', 120):
            if FLAGS.mode == 'async_valid':
              melt.mark_evaluated_model(model_path_)
            _write_metric_hours_file(log_dir, step)
            try:
              _write_metric_hours_file(log_dir, step, is_base=True)
            except Exception:
              pass
          
          # if FLAGS.sync_hdfs and FLAGS.ori_log_dir and log_dir and (log_dir != FLAGS.ori_log_dir):
          #   # command = "rsync -a --update  --exclude 'model*' --exclude 'ckpt*' %s %s &" % (log_dir, os.path.dirname(FLAGS.ori_log_dir))
          #   command = f'scp -r {log_dir}/*metric* {FLAGS.ori_log_dir} &'
          #   gezi.system(command)
          #   command = f'scp -r {log_dir}/*flags* {FLAGS.ori_log_dir} &'
          #   gezi.system(command)

          # if FLAGS.valid_hour and (FLAGS.ori_log_dir != FLAGS.log_dir) and FLAGS.sync_hdfs:
          #   command = f'mkdir -p {FLAGS.ori_log_dir}/infos/{FLAGS.valid_hour}'
          #   gezi.system(command)
          #   command = f'{sync} {log_dir}/infos/{FLAGS.valid_hour}/* {FLAGS.ori_log_dir}/infos/{FLAGS.valid_hour} &'
          #   gezi.system(command)

        # for normal valid --mode=valid will not write summary !
        if FLAGS.mode != 'async_valid':
           melt.write_metric_summaries(names, vals, step)
           melt.save_eval_step(step)
        else:
          one_step = 1
          if FLAGS.loop_type == 'day':
            one_step = 24 * FLAGS.valid_interval_epochs
            assert one_step == int(one_step)
          step2 = None
          wait_span = 10
          times = 0
          timeout_times = int(FLAGS.eval_timeout / wait_span)
          while True:
            step2 = melt.get_eval_step(from_file=True)
            if step2 + one_step >= step:
              break
            logging.info(f'{FLAGS.valid_hour}: read step from file {step2} + {one_step} != {step}, wating (times:{times}/{timeout_times})... {FLAGS.log_dir}/eval_step.txt')
            time.sleep(wait_span)
            times += 1
            if times == timeout_times:
              logging.info(f'{FLAGS.valid_hour}: read step from file {step2} + {one_step} != {step}, wating (times:{times})... timeout ignore and contiue')
              break
          
          write_summary = False
          step2 = melt.get_eval_step(from_file=True)
          if not FLAGS.loop_train and not FLAGS.from_loop_train:
            step += 1
          if step2 + one_step <= step:
            write_summary = True
          else:
            logging.info(f'read step from eval_step.txt {step2} + 1 > {step}, will not write summary')
          logging.info(f'{FLAGS.valid_hour} {step} write_summary:{write_summary}')

          if write_summary:
            # os.system(f'mkdir -p {FLAGS.log_dir}/eval_steps/{step}')
            with gezi.FileLock(f'{FLAGS.log_dir}/summary.lock'):
              gezi.append_to_txt(f'{FLAGS.valid_hour}:{step}', f'{FLAGS.log_dir}/summary_steps.txt')
              melt.write_metric_summaries(names, vals, step)
              melt.save_eval_step(step)
              time.sleep(1)

          deleted = False
          if FLAGS.del_inter_model and FLAGS.del_model_path:
            wait_span = 10
            times = 0
            timeout_times = int(FLAGS.del_timeout / wait_span)
            need_wait = melt.is_evaluating_model(FLAGS.del_model_path)
            while True:
              is_evaluated = melt.is_evaluated_model(FLAGS.del_model_path) \
                  or os.path.exists(f'{FLAGS.log_dir}/infos/{FLAGS.valid_hour}/metrics.csv') \
                  or os.path.exists(f'{FLAGS.log_dir}/infos/{FLAGS.valid_hour}/valid.csv') \
                  or os.path.exists(f'{FLAGS.log_dir}/infos/{FLAGS.valid_hour}/no_metrics.csv')
              if is_evaluated or not need_wait:
                deleted = True
                if FLAGS.del_model_path != melt.latest_checkpoint(model_dir):
                  command = f'rm -rf {FLAGS.del_model_path}*' 
                  logging.info(f'{command} new model is {melt.latest_checkpoint(model_dir)}')
                  os.system(command)
                break
              logging.info(f'{FLAGS.del_model_path} has not finished evaluation, waiting (times:{times}/{timeout_times})')
              time.sleep(wait_span)
              times += 1
              if times == timeout_times:
                logging.info(f'{FLAGS.valid_hour}, wating (times:{times})... timeout force to delete {FLAGS.del_model_path} and contiue')
                break
          logging.info(f'{FLAGS.valid_hour} {step} deleted:{deleted}')
          if not deleted:
            if FLAGS.del_model_path != melt.latest_checkpoint(model_dir):
              command = f'rm -rf {FLAGS.del_model_path}*' 
              logging.info(f'{command} new model is {melt.latest_checkpoint(model_dir)}')
              os.system(command)
          
          _try_eval_day()

    exit(0)

def _on_epoch_end(model_dir, log_dir, save_model=True, del_model_path=None):   
  use_horovod = FLAGS.use_horovod
  if use_horovod:
    hvd = _import_horovod()
  rank = FLAGS.local_rank
  if rank == 0:  
    # if del_model_path and os.path.exists(os.path.join(FLAGS.log_dir, 'step.txt')):
    latest_checkpoint = melt.latest_checkpoint(model_dir)
    if del_model_path and FLAGS.del_inter_model and save_model and os.path.basename(del_model_path) != os.path.basename(latest_checkpoint):
      if not FLAGS.metric_eval or FLAGS.train_only or not FLAGS.async_valid:
        # if os.path.exists(del_model_path):
        if del_model_path != melt.latest_checkpoint(model_dir):
          command = f'rm -rf {del_model_path}*' 
          logging.debug(command + ' ' + 'new model is ' + melt.latest_checkpoint(model_dir))
          os.system(command)
    
    if FLAGS.async_valid:
      melt.mark_evaluating_model(latest_checkpoint)

    if FLAGS.del_inter_events:
      command = 'rm -rf {log_dir}/events*'
      gezi.system(command)

    # if you set ori_model_dir then will sync to it here or if you use CloudS path then HACK
    # TODO all of these shoud be in defualt callbacks so to make flow.py cleaner
    if model_dir:
      if not FLAGS.train_hour:
        # command = 'rsync -a --update --delete %s %s &' % (model_dir, os.path.dirname(FLAGS.ori_model_dir))
        # gezi.system(command)
        pass
      else:
        if (FLAGS.train_hour.endswith('23') or FLAGS.train_hour.endswith('28')) and FLAGS.sync_hdfs:
          if FLAGS.ori_model_dir != FLAGS.model_dir:
            command = f'rm -rf {FLAGS.ori_model_dir}/model.ckpt*'
            gezi.system(command)
          else:
            command = f'mkdir -p {FLAGS.ori_model_dir}/models/{FLAGS.train_hour}'
            gezi.system(command) 
            latest_model = melt.latest_checkpoint(model_dir)
            command = f'{sync} {latest_model}* {FLAGS.ori_model_dir}/models/{FLAGS.train_hour} &'
            gezi.system(command)
            command = f'{sync} {model_dir}/checkpoint* {FLAGS.ori_model_dir}/models/{FLAGS.train_hour} &'
            gezi.system(command)
            command = f'{sync} {latest_model}* {FLAGS.ori_model_dir} &'
            gezi.system(command)
            command = f'{sync} {model_dir}/checkpoint {FLAGS.ori_model_dir} &'
            gezi.system(command)

    # TODO FIXME 非loop train 模式，async valid 产出metric hours但是没有写metric 到summary 只写了其它summary 
    # read step from eval_step.txt 0 + 1 > 0, will not write summary 哪里少了 inc_eval_step? 和 loop train区别在？
    if FLAGS.valid_hour and model_dir:
      gstep = melt.get_eval_step()
      if not FLAGS.valid_input:
        gstep = melt.inc_eval_step()
      pre_model_path = os.path.basename(del_model_path) if del_model_path else 'None'
      try:
        hour_info = f'{FLAGS.valid_hour}\t{gstep}' \
                    + '\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) \
                    + '\t' + os.path.basename(melt.latest_checkpoint(model_dir)) \
                    + '\t' + pre_model_path
        gezi.append_to_txt(hour_info, os.path.join(model_dir, 'valid_hours.txt'))
      except Exception:
        pass

    if not FLAGS.async_valid and FLAGS.ev_last:
      if FLAGS.valid_input and (FLAGS.metric_values or gezi.get('metric_values')) and model_dir:
        gstep = melt.get_eval_step()
        _write_metric_file(model_dir, gstep)
        _write_metric_hours_file(model_dir, gstep)
        _write_metric_file(model_dir, gstep, is_base=True)
        _write_metric_hours_file(model_dir, gstep, is_base=True)
          
      if FLAGS.sync_hdfs and FLAGS.ori_log_dir and log_dir and (log_dir != FLAGS.ori_log_dir):
        # command = "rsync -a --update  --exclude 'model*' --exclude 'ckpt*' %s %s &" % (log_dir, os.path.dirname(FLAGS.ori_log_dir))
        if FLAGS.metric_eval:
          command = f'scp -r {log_dir}/*metric* {FLAGS.ori_log_dir}'
          gezi.system(command)
          command = f"scp {log_dir}/*flags* {FLAGS.ori_log_dir}"
          gezi.system(command)
            
          if FLAGS.valid_hour:
            command = f'mkdir -p {FLAGS.ori_log_dir}/infos/{FLAGS.valid_hour}'
            gezi.system(command)
            command = f'{sync} {log_dir}/infos/{FLAGS.valid_hour}/* {FLAGS.ori_log_dir}/infos/{FLAGS.valid_hour} &'
            gezi.system(command)

      _try_eval_day()

def get_session(sess):
  """get real session"""
  session = sess
  while type(session).__name__ != 'Session':
      # pylint: disable=W0212
      session = session._sess
  return session

def tf_train_flow(train_once_fn, 
                  model_dir=None,
                  log_dir=None, 
                  max_models_keep=1, 
                  save_interval_seconds=600, 
                  save_interval_steps=1000, 
                  num_epochs=None,
                  num_steps=None, 
                  save_model=True,
                  save_interval_epochs=None, 
                  freeze_graph=False,
                  num_steps_per_epoch=0,
                  restore_from_latest=True,
                  metric_eval_fn=None,
                  valid_interval_epochs=0,
                  first_interval_epoch=-1.,
                  inference_fn=None, 
                  inference_interval_epochs=0,
                  init_fn=None,
                  restore_fn=None,
                  restore_include=None,
                  restore_exclude=None,
                  save_all_scope=False,  # TODO save load from restore scope only but svae all
                  variables_to_restore=None,
                  variables_to_save=None,  # by default will be the same as variables_to_restore
                  output_collection_names=None, 
                  output_node_names=None,
                  learning_rate=None,  # not use yet, just use in train_once
                  learning_rate_patience=None,
                  learning_rate_decay_factor=None,
                  write_during_train=True,
                  model=None,
                  callbacks=[],
                  sess=None):
  """
  similary flow as tf_flow, but add model try reload and save
  """
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd
  rank = 0 
  if use_horovod:
    rank = hvd.rank()

  model_dir_ = model_dir
  if rank != 0:
    model_dir = None

  if not FLAGS.metric_eval:
    metric_eval_fn = None

  if FLAGS.ps_strategy and FLAGS.round == 0:
    server = gezi.get('server')
    sess  = tf.compat.v1.train.MonitoredTrainingSession(master=server.target,
                                                        is_chief=(FLAGS.task_index == 0),
                                                        #checkpoint_dir=FLAGS.model_dir
                                                       )
    sess.graph._unsafe_unfinalize()
    save_model = False
  
  if sess is None:
    #TODO melt.get_session is global session but may cause non close at last
    sess = melt.get_session()

  if FLAGS.use_tpu:
    sess.run(tpu.initialize_system())

  if model_dir:
    if model:
      # here have not saved optimizer checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
      checkpoint = tf.train.Checkpoint(model=model)
      ckpt_dir = model_dir + '/ckpt'
      checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')
  
    assert tf.__version__ < '2', 'tf 2+ not support slim, TODO remove slim dependency'
    ## be caefull slim.get_variables_to_restore include and exlcude means top scope actually you need to pass wide_deep/deep/doc_emb instead of only pass doc_emb
    if not variables_to_restore:
      # variables_to_restore = slim.get_variables_to_restore(include=restore_include, exclude=restore_exclude)
      variables_to_restore = slim.get_variables_to_restore(include=None, exclude=None)
    
    # logging.debug('restore_include', restore_include, 'restore_exclude', restore_exclude)
    # logging.debug('variables_to_restore', variables_to_restore)

    # TODO need to set variables to save otherwise will save dataset and fail FIXME
    if not variables_to_save:
      variables_to_save = variables_to_restore
    if save_all_scope:
      variables_to_save = None
    
    #load all var in checkpoint try to save all var(might more then original checkpoint) if not specifiy variables_to_save
    varnames_in_checkpoint = melt.get_checkpoint_varnames(model_dir)
    #logging.info('varnames_in_checkpoint: {}'.format(varnames_in_checkpoint))
    
    # TODO has someproblem say  tf.Variable 'r_net/text_encoder/cudnn_rnn/cu_dnngru/recurrent_kernel/adam_v:0' even though in checkpoint I have renated it as ignore/rnet
    # TODO tf2 graph mode but could not use slim 
    variables_to_restore_from_model = slim.get_variables_to_restore(include=varnames_in_checkpoint)
    #logging.info('variables_to_restore_from_model: {}'.format(variables_to_restore_from_model))
    if not variables_to_restore:
      variables_to_restore = variables_to_restore_from_model
    else:
      variables_to_restore = [v for v in variables_to_restore if v in variables_to_restore_from_model]

    # TODO add regex patter exlucde include
    if restore_exclude:
      def _exclude_ok(name, restore_exclude):
        for excl in restore_exclude:
          if excl in name:
            return False
        return True
      variables_to_restore = [v for v in  variables_to_restore if _exclude_ok(v.name, restore_exclude)]
    if restore_include:
      def _include_ok(name, restore_include):
        for incl in restore_include:
          if incl in name:
            return True
        return False
      variables_to_restore = [v for v in  variables_to_restore if _include_ok(v.name, restore_include)]
        
    #--tf 1.6 adadelta will have same vars... 
    variables_to_restore = list(set(variables_to_restore))
    #logging.info('variables_to_restore', variables_to_restore[:100])
    logging.debug('variables_to_restore(not show Optimize):\n', '\n'.join([f'{x}' for x in variables_to_restore if not 'OptimizeLoss' in x.name][:100]))

  global_step = tf.compat.v1.train.get_or_create_global_step()

  loader = tf.compat.v1.train.Saver(var_list=variables_to_restore) 

  logging.debug('max models to keep {}, keep every {} hours'.format(max_models_keep, save_interval_seconds / 3600.0))
  saver = tf.compat.v1.train.Saver(
    max_to_keep=max_models_keep, 
    keep_checkpoint_every_n_hours=save_interval_seconds / 3600.0,
    var_list=variables_to_save) 
  epoch_saver = tf.compat.v1.train.Saver(var_list=variables_to_save, max_to_keep=max_models_keep)
  best_epoch_saver = tf.compat.v1.train.Saver(var_list=variables_to_save) 
  dist_saver = tf.compat.v1.train.Saver(var_list=variables_to_save, sharded=True, max_to_keep=1)

  init_op = tf.group(tf.compat.v1.global_variables_initializer(), #variables_initializer(global_variables())
                     tf.compat.v1.local_variables_initializer()) #variables_initializer(local_variables())

  timer = gezi.Timer('sess run init_op in melt.tf_train_flow')
  #model.save('./weights')
  # notice 
  
  init_iters(sess)
  if FLAGS.round == 0:
    sess.run(init_op)
  timer.print_elapsed()

  if model is not None and FLAGS.round == 0:
    if hasattr(model, 'init'):
      model.init()
    if init_fn:
      try:
        init_fn(model)
      except Exception:
        pass
    if hasattr(model, 'restore'):
      model.restore() 
    if restore_fn:
      try:
        restore_fn(model)
      except Exception:
        pass
    
    print_fn = logging.info if FLAGS.round == 0 and FLAGS.work_mode == 'train' else logging.debug
    melt.print_model(model, print_fn=print_fn, depth=FLAGS.print_depth)

  #pre_step means the step last saved, train without pretrained,then -1
  pre_step = -1
  fixed_pre_step = -1  #fixed pre step is for epoch num to be correct if you change batch size
  #print(model_dir)
  pre_epoch = None
  del_model_path = None
  if model_dir:
    # TODO refactor
    model_path = _get_model_path(model_dir, save_model)

    model_dir = gezi.get_dir(model_dir) #incase you pass ./model/model-ckpt1000 -> ./model

    # for train_loop only load model from round 0
    if model_path is not None:
      if not restore_from_latest:
        logging.info('using recent but not latest model')
        model_path = melt.recent_checkpoint(model_dir)
      model_name = os.path.basename(model_path)
      
      if FLAGS.model_path:
        model_path = FLAGS.model_path
      if FLAGS.work_mode == 'train':
        FLAGS.del_model_path = model_path

      if FLAGS.round == 0:
        timer = gezi.Timer('Loading from existing model [%s]' % model_path, print_fn=logging.info)
        if restore_fn is not None:
          restore_fn(sess)
        loader.restore(sess, model_path)
        timer.print()

      # pre_step = melt.get_model_step(model_path) - 1 if FLAGS.global_step is None else FLAGS.global_step -1
      pre_step = sess.run(tf.compat.v1.train.get_global_step()) - 1 if FLAGS.global_step is None else -1
      pre_epoch = melt.get_model_epoch(model_path) if FLAGS.global_epoch is None else FLAGS.global_epoch
      fixed_pre_step = pre_step
      del_model_path = model_path
    else:
      latest_checkpoint = None
      if not use_horovod: # now will hang
        try:
          latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
          if FLAGS.model_path:
            latest_checkpoint = FLAGS.model_path
          if latest_checkpoint:
            if FLAGS.round == 0:
              logging.info('Try start from eager trained mode, latest checkpoint:', latest_checkpoint)
              checkpoint.restore(latest_checkpoint).run_restore_ops(session=sess)

            pre_epoch = int(latest_checkpoint.split('-')[-1])

            pre_step = sess.run(tf.compat.v1.train.get_global_step()) - 1
            fixed_pre_step = pre_step
            logging.info('Start step is:', pre_step)
        except Exception:
          logging.info('Something wrong with restore from eager trained model')
        if latest_checkpoint is None:
          logging.info('Train all start step 0')
 
          if FLAGS.round == 0:
            if init_fn is not None:
              init_fn(sess)
      del_model_path = latest_checkpoint

  if FLAGS.local_rank != 0:
    model_dir = None
    del_model_path = None
    save_model = False

  try:
    learning_rate = tf.compat.v1.get_collection('learning_rate')[-1]
    learning_rate_weight = tf.compat.v1.get_collection('learning_rate_weight')[-1]
    sess.run(tf.compat.v1.assign(learning_rate, learning_rate * learning_rate_weight))
  except Exception:
    # if not using weight_decay but using optimizer decay then will go here as learning rate is a tensor can not assign
    pass
 
  try:
    logging.info('Actual start global step:', sess.run(global_step), 'learning rate:', sess.run(learning_rate), 'learning_rate_weight:', sess.run(learning_rate_weight))
  except Exception:
    pass
  
  if FLAGS.work_mode == 'train' and FLAGS.metric_eval and FLAGS.monitor_l2:
    # l2 consuming 0.75 s
    total_params = sess.run(tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in tf.compat.v1.trainable_variables()]))
    l2 = sess.run(tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])) / total_params
    # total_params = 1
    # l2 = 0.
    logging.debug('Model total training parameters is:', total_params, 'with initial l2:', l2)
    FLAGS.l2_ = l2
    FLAGS.params_ = total_params

  if FLAGS.round == 0:
    if use_horovod:
      bcast = hvd.broadcast_global_variables(0)
      sess.run(bcast) 

  if model_dir_:
    #if save_interval_epochs and num_steps_per_epoch and num_steps >= 0:
    epoch_dir = os.path.join(model_dir_, 'epoch')
    if rank == 0:
      gezi.try_mkdir(epoch_dir)
    checkpoint_path = os.path.join(model_dir_, 'model.ckpt')

  #tf.train.write_graph(sess.graph_def, model_dir, 'train.pbtxt')
  only_one_step = False
  
  if use_horovod:
    comm = gezi.get_global('dist').comm
    ## TODO FIXME why bcast here not work ? simple test work see tests/bcast.py
    #comm.bcast(pre_step, root=0)
    temp = np.array([pre_step, fixed_pre_step])
    comm.Bcast(temp, root=0)
    pre_step, fixed_pre_step = temp[0], temp[1]

  step = start = pre_step + 1
  fixed_step = fixed_pre_step + 1 

  #hack just for save one model after load
  if num_steps < 0 or (num_steps and num_steps < step):
    logging.info('just load and resave then exit, -1 means save model and pb, -2 means only save model, -3 means only save pb')
    model_path_ =  _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch, epoch=pre_epoch)
    if num_steps != -3:
      saver.save(sess, model_path_, global_step=step + 1)
    # if freeze_graph:
    # melt.freeze_graph(sess, model_path_, step + 1, output_collection_names, output_node_names)
    if num_steps != -2:
      melt.freeze_graph(sess, os.path.join(model_dir, 'model'), None, output_collection_names, output_node_names)
    sess.close()
    exit(0)
  
  if num_epochs < 0:
    only_one_step = True
    logging.info('just run one step')

  _try_eval(model_dir_, log_dir, metric_eval_fn, inference_fn)

  #early_stop = True #TODO allow config
  num_bad_epochs = 0
  pre_epoch_eval_loss = 1e20
  best_epoch_eval_loss = 1e20
  num_allowed_bad_epochs = 4 #allow 5 non decrease eval loss epochs  before stop
  epoch_saved_step = 0
  num_epochs = num_epochs if num_epochs else 1024

  for callback in callbacks:
    if hasattr(callback, 'set_model'):
      callback.set_model(model)

  #-------------------------------main loop
  timer_, FLAGS.total_time, FLAGS.train_time, FLAGS.valid_time = gezi.Timer(reset=False), None, None, None
  try:
    # num_epochs + 1 safe, since we need one more step to do final evaluation, and we can escape for loop, when step all done
    start_epoch = FLAGS.start_epoch or int(step / num_steps_per_epoch) if not FLAGS.train_loop else 0
    if FLAGS.round > 0:
      step = 0
      fixed_step = 0
    if not FLAGS.train_loop and (num_epochs > int(num_epochs)):
      end_epoch = start_epoch + int(num_epochs) + 1
    else:
      end_epoch = start_epoch + int(num_epochs)
    # epoch单独一个loop train 内部loop 这样 train和 async eval 的 valid loop的屏幕打印能分两行显示不干扰
    for epoch in tqdm(range(start_epoch, end_epoch), desc='Training', ascii=True):
      logging.debug('------------------------epoch:', epoch)
      for callback in callbacks:
        if hasattr(callback, 'on_epoch_begin'):
          kwargs = {}
          if 'lr' in inspect.getargspec(callback.on_epoch_begin).args:
            kwargs['lr'] = learning_rate
          callback.on_epoch_begin(epoch, **kwargs)


      train_hour = FLAGS.train_hour if FLAGS.loop_train else None
      desc = 'Epoch:%2d/%d' % (epoch + 1, int(num_epochs)) if not train_hour else '%s-%d/%d Epoch:%2d/%d' % (train_hour, FLAGS.round + 1, FLAGS.num_rounds, epoch + 1, int(num_epochs))
      t = tqdm(range(num_steps_per_epoch), total=num_steps_per_epoch, desc=desc, ascii=True)
      for i in t:
        gstep = sess.run(global_step)
        step = int(gstep)
        # if i % 10 == 0:
        #   print(step, i, FLAGS.task_index)
        if step >= num_steps_per_epoch * (epoch + 1):
          break
        
        postfix = {}
        if gezi.get('loss'):
          postfix['loss'] = gezi.get('loss')
        if gezi.get('valid_loss'):
          postfix['val_loss'] = gezi.get('valid_loss')
        t.set_postfix(postfix)
        model_step_path = None
        if model_dir_:
          model_path_ = os.path.join(epoch_dir,'model.ckpt-%.2f'%((fixed_step + 1)  / float(num_steps_per_epoch)))
          model_step_path_ = model_path_ + '-' + str(step + 1)
          if (write_during_train and metric_eval_fn is not None and valid_interval_epochs > 0 \
              and (fixed_step + 1) % int(num_steps_per_epoch * valid_interval_epochs) == 0 \
              or first_interval_epoch > 0 \
                and (fixed_step + 1) == int(num_steps_per_epoch * first_interval_epoch)):
                # and (fixed_step == int(num_steps_per_epoch * first_interval_epoch)) or \
                #      fixed_step == int(num_steps_per_epoch * (1 + first_interval_epoch))):
            model_step_path = model_step_path_
          else:
            model_step_path = None

        # if step == 0:
        #   model_step_path = None

        for callback in callbacks:
          if hasattr(callback, 'on_batch_begin'):
            kwargs = {}
            if 'lr' in inspect.getargspec(callback.on_batch_begin).args:
              kwargs['lr'] = learning_rate
            callback.on_batch_begin(step, **kwargs)

        #print('--------------------step', step)
        stop = train_once_fn(sess, 
                             step, 
                             is_start=(step==start), 
                             fixed_step=fixed_step,
                             num_epochs=num_epochs,
                             model_path=model_step_path,
                             use_horovod=use_horovod,
                             valid_interval_epochs=valid_interval_epochs,
                             timer_=timer_,
                             ## TODO FIXME this line will cause   tensorflow.python.framework.errors_impl.NotFoundError: Resource localhost/save_counter/N10tensorflow3VarE does not exist. 
                            )

        if only_one_step:
          stop = True

        step += 1
        fixed_step += 1

        for callback in callbacks:
          if hasattr(callback, 'on_batch_end'):
            kwargs = {}
            if 'lr' in inspect.getargspec(callback.on_batch_end).args:
              kwargs['lr'] = learning_rate
            callback.on_batch_end(step, **kwargs)

        # Already inited in melt.apps.train
        #if step == 1 and model is not None and hasattr(model, 'init_predict'):
        #  model.init_predict()

        if save_model and step and model_dir:
          #step 0 is also saved! actually train one step and save
          is_step_save = step % save_interval_steps == 0
          is_epoch_save = FLAGS.save_interval_epochs and FLAGS.save_interval_epochs > 0 \
              and save_interval_steps and num_steps_per_epoch and fixed_step % int(num_steps_per_epoch * save_interval_epochs) == 0 \
              and not (num_steps_per_epoch * num_epochs - step < int(num_steps_per_epoch * save_interval_epochs))

          is_step_save = is_step_save or is_epoch_save 

          if is_step_save:
            model_path_ = _get_checkpoint_path(checkpoint_path, fixed_step, num_steps_per_epoch)
            timer = gezi.Timer('save model step %d to %s'%(step, checkpoint_path), False)
            if rank == 0:
              saver.save(sess, model_path_, global_step=step)
              if freeze_graph:
                melt.freeze_graph(sess, model_path_, step, output_collection_names, output_node_names)
            
            # if FLAGS.local_mark in log_dir and FLAGS.sync_hdfs and (rank == 0):
            #   command = f"rsync -a --update  --exclude 'model*' --exclude 'ckpt*' %s %s &" % (log_dir, os.path.dirname(FLAGS.ori_log_dir))
            #   gezi.system(command)
            
            timer.print_elapsed()
    
          if is_epoch_save:
            epoch_saved_step = step
            if rank == 0:
              if FLAGS.async_valid and FLAGS.valid_input:
                FLAGS.total_time = (time.time() - gezi.get_global('start_time')) / 60
                _async_valid()
            #logging.info(timer.elapsed())
            timer.print_elapsed()
              
            if freeze_graph:
              melt.freeze_graph(sess, model_path_, step, output_collection_names, output_node_names)

            # TODO FIXME if add keras save below wil hang, might due to rank 0 save so slower then others(conflict with evaluate)
            # if not use evaluate just train + save ok... sitll not find reason...
            # Add comm.barrier below but still might hang, though not hang on first time, so not save using horovod
            # [1,0]<stderr>:Stalled ranks:
            # [1,0]<stderr>:0: [HorovodAllreduce_Const_9_0]
            # seems ok move it after freeze_graph
            if model and not use_horovod and FLAGS.save_eager_ckpt:
            #if model:          
              #model.save_weights(epoch_dir + '/ckpt-%.2f' % (fixed_step / float(num_steps_per_epoch)))
              # TODO FIXME if restart will save from 1... again..
              timer = gezi.Timer('keras epoch save to {}'.format(checkpoint_prefix))
              checkpoint.save(checkpoint_prefix, session=sess)
              #print(sess.run(checkpoint.save_counter))
              #logging.info(timer.elapsed())
              timer.print_elapsed()

          # if write_during_train:
          #   if inference_fn is not None and inference_interval_epochs and fixed_step % int(num_steps_per_epoch * inference_interval_epochs) == 0:
          #     model_path_ = os.path.join(epoch_dir,'model.ckpt-%.2f' % (fixed_step / float(num_steps_per_epoch)))
          #     model_step_path = model_path_ + '-' + str(step)
          #     try:
          #       inference_fn(model_path=model_step_path)
          #     except Exception:
          #       logging.warning(traceback.format_exc())  
          
        if stop is True:
          print('Early stop running %d stpes'%(step), file=sys.stderr)
          raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
        if num_steps and (step + 1) == start + num_steps:
          raise tf.errors.OutOfRangeError(None, None,'Reached max num steps')
        #max_num_epochs = 1000
        max_num_epochs = num_epochs
        #if max_num_epochs and num_steps_per_epoch and fixed_step // num_steps_per_epoch >= max_num_epochs:
        if max_num_epochs and num_steps_per_epoch and fixed_step / num_steps_per_epoch > max_num_epochs:
          raise tf.errors.OutOfRangeError(None, None,'Reached max num epochs of %d'%max_num_epochs)

        # TODO might change learning rate here ?
        for callback in callbacks:     
          if hasattr(callback, 'on_epoch_end'):
            kwargs = {}
            if 'lr' in inspect.getargspec(callback.on_epoch_end).args:
              kwargs['lr'] = learning_rate
            callback.on_epoch_end(epoch, **kwargs)
    if FLAGS.ps_strategy and FLAGS.local_rank == 0:
      model_path_ = _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch)
      # if you want to evaluate at last just set valid_interval_epochs=1 or 0.5 0.25 0.2 0.1
      dist_saver.save(get_session(sess), model_path_, global_step=step)
    raise tf.errors.OutOfRangeError(None, None, 'Reached max num epochs of %d' % max_num_epochs)
  except tf.errors.OutOfRangeError:
    if rank == 0:
      melt.inc_total_step(int(num_steps_per_epoch * num_epochs))
      if (step - epoch_saved_step > 1) and not (step==start) and save_model and step % save_interval_steps != 0 and model_dir:
        model_path_ = _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch)
        # if you want to evaluate at last just set valid_interval_epochs=1 or 0.5 0.25 0.2 0.1
        saver.save(sess, model_path_, global_step=step)
        if FLAGS.async_valid and FLAGS.valid_input and FLAGS.ev_last:
          FLAGS.total_time = (time.time() - gezi.get_global('start_time')) / 60
          _async_valid()
        if not use_horovod and FLAGS.save_eager_ckpt:
          checkpoint_prefix = os.path.join(model_dir, 'ckpt', 'ckpt.final')
          checkpoint.save(checkpoint_prefix, session=sess)

      #if freeze_graph:
      if save_model and FLAGS.freeze_graph_final:
        if (FLAGS.round == FLAGS.num_rounds - 1):
          melt.freeze_graph(sess, os.path.join(model_dir, 'model'), None, output_collection_names, output_node_names)
        
    # hack for hvd we store the last keras checkpoint
    if use_horovod and hvd.rank() == 0 and FLAGS.save_eager_ckpt:
      checkpoint_prefix = os.path.join(model_dir, 'ckpt', 'ckpt.final')
      checkpoint.save(checkpoint_prefix, session=sess)
    
    if only_one_step:
      # TODO strange logging.info will not show to screen if using horovod
      logging.info('Done one step')
      exit(0)
    
    if (num_epochs and fixed_step / num_steps_per_epoch >= num_epochs) or (num_steps and step == start + num_steps) :
      logging.info('Done training for %.3f epochs, %d steps.' % (fixed_step / num_steps_per_epoch, step))
      #FIXME becase coord.join seems not work,  RuntimeError: Coordinator stopped with threads still running: Thread-9
      # exit(0)
    else:
      logging.info('Should not stop, but stopped at epoch: %.3f'%(fixed_step / num_steps_per_epoch))
      logging.info(traceback.format_exc())
      #raise e

    FLAGS.total_time = (time.time() - gezi.get_global('start_time')) / 60
    logging.info(f'Round:{FLAGS.round} Train:{FLAGS.train_hour} Valid:{FLAGS.valid_hour}', 'TotalTime:{:.1f}m'.format(FLAGS.total_time))
    _on_epoch_end(model_dir, log_dir, save_model, del_model_path)

    if inference_fn is not None:
      inference_fn(FLAGS.model_dir)

  if not FLAGS.train_loop:
    if FLAGS.use_tpu:
      sess.run(tpu.shutdown_system())
    sess.close()
  else:
    logging.info(f'Done for {FLAGS.train_input} round:{FLAGS.round}')

#@TODO not tested yet
def tf_test_flow(test_once, model_dir='./model', 
                 model_name=None, num_epochs=1, num_steps=0,
                 sess=None):
  """
  basic flow for tf records, allow most freedom for usage, if not tfrecords no need for flow
  Args:
  test_once: function with 2 inputs sess and step
  model_dir: can be dir like ./model will fetch lates model in model dir , or be real model path like ./model/model.0.ckpt
  """
  if sess is None:
    sess = tf.compat.v1.InteractiveSession()

  melt.restore(sess, model_dir, model_name)

  if not os.path.isdir(model_dir):
    model_dir = os.path.dirname(model_dir)
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(model_dir, sess.graph)

  # coord = tf.train.Coordinator()
  # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  try:
    step = 0
    #while not coord.should_stop():
    while True:
      test_once(sess, step)
      step += 1
      if num_steps and step == num_steps:
        raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  except tf.errors.OutOfRangeError:
    print('Done testing for %d epochs, %d steps.' % (num_epochs, step))
