#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 12:59:29.331219
#   \Description  
# ==============================================================================

"""
@TODO better logging, using logging.info ?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six 
if six.PY2:
  from io import BytesIO as IO
else:
  from io import StringIO as IO 

import sys, os, traceback
import inspect

import gezi 
logging = gezi.logging

import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline

from tensorflow.keras import backend as K
import numpy as np
import math

import gezi
from gezi import Timer, AvgScore 
import melt

try:
  projector_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
except Exception:
  pass

def profile_step(profiler, step, run_metadata, timeline_dir):
  profiler.add_step(step=step, run_meta=run_metadata)
  
  profile_opt_builder = option_builder.ProfileOptionBuilder()

  #过滤条件：只显示排名top 5
  profile_opt_builder.with_max_depth(30) \
    .with_min_execution_time(min_micros=1000) \
    .select(['micros','occurrence']).order_by('micros')

  #显示视图为op view
  profiler.profile_operations(profile_opt_builder.build())

  # profiler.profile_name_scope(profile_opt_builder.build())
  # profiler.profile_graph(profile_opt_builder.build())   
  # profile_opt_builder = option_builder.ProfileOptionBuilder()

  # # # #过滤条件：显示minist.py代码。
  profile_opt_builder.with_max_depth(1000)
  #  TODO why not work, adding will filter all..
  profile_opt_builder.with_node_names(show_name_regexes=['model.py.*'])
  # #显示视图为code view
  profiler.profile_python(profile_opt_builder.build())

  profiler.advise(options=model_analyzer.ALL_ADVICE)
  
  # Write Timeline data to a file for analysis later
  tl = timeline.Timeline(run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format()
  with open(f'{timeline_dir}/{step}.json', 'w' ) as f:
    f.write(ctf)
  if step / FLAGS.profile_interval_steps == 2:
    exit(0)

def train_once(sess, 
               step, 
               ops, 
               names=None,
               gen_feed_dict_fn=None, 
               deal_results_fn=None, 
               interval_steps=100,
               eval_ops=None, 
               eval_names=None, 
               gen_eval_feed_dict_fn=None, 
               deal_eval_results_fn=None, 
               valid_interval_steps=100, 
               valid_interval_epochs=1.,
               print_time=True, 
               print_avg_loss=True, 
               model_dir=None, 
               log_dir=None, 
               is_start=False,
               num_steps_per_epoch=None,
               metric_eval_fn=None,
               metric_eval_interval_steps=0,
               summary_excls=None,
               fixed_step=None,   # for epoch only, incase you change batch size
               eval_loops=1,
               learning_rate=None,
               learning_rate_patience=None,
               learning_rate_decay_factor=None,
               num_epochs=None,
               model_path=None,
               use_horovod=False,
               timer_=None,
               ):

  # print(ops, eval_ops, valid_interval_steps)
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd

  rank = 0
  if use_horovod:
    rank = hvd.rank()

  if step == 0:
    melt.print_summary_ops()
    # logging.debug(tf.get_collection(tf.GraphKeys.TRAIN_OP))
    # logging.debug(ops, eval_ops)
    # logging.debug(tf.global_variables())

  timer = gezi.Timer()
  if not hasattr(train_once, 'timer'):
    train_once.timer = Timer()
    train_once.eval_timer = Timer()
    train_once.metric_eval_timer = Timer()
  
  if step == 0:
    train_once.timer.step = 0

  if log_dir:
    if not hasattr(train_once, 'summary_op'):
      melt.print_summary_ops()
      if summary_excls is None:
        train_once.summary_op = tf.compat.v1.summary.merge_all()
      else:
        summary_ops = []
        for op in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES):
          for summary_excl in summary_excls:
            if not summary_excl in op.name:
              summary_ops.append(op)
        for op in summary_ops:
          logging.debug('filtered summary_op:', op)
        train_once.summary_op = tf.compat.v1.summary.merge(summary_ops)
      
      summary_dir = os.path.join(model_dir, 'main') if FLAGS.train_valid_summary else log_dir
      train_once.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir, sess.graph)
      if FLAGS.train_valid_summary:
        train_once.train_writer = gezi.SummaryWriter(os.path.join(log_dir, 'train'))
        train_once.valid_writer = gezi.SummaryWriter(os.path.join(log_dir, 'valid'))
      else:
        train_once.train_writer, train_once.valid_writer = None, None

      try:
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_once.summary_writer, projector_config)
      except Exception:
        pass
   
  melt.set_global('step', step)
  epoch = (fixed_step or step) / num_steps_per_epoch if num_steps_per_epoch else -1
  if not num_epochs:
    epoch_str = 'epoch:%.3f' % (epoch) if num_steps_per_epoch else ''
  else:
    epoch_str = 'epoch:%.3f/%d' % (epoch, num_epochs) if num_steps_per_epoch else ''
  melt.set_global('epoch', '%.2f' % (epoch))

  if FLAGS.train_hour:
    epoch_str = f'train_hour:{FLAGS.train_hour} {epoch_str}'
  
  info = IO()
  stop = False
  summary_str = []

  step_ = step + 1
  is_first_steps = step_ == 1 or step_ == 100 or step_ == 200
  is_last_step = step_ == num_steps_per_epoch
  is_interval_step = interval_steps and (step_ % interval_steps == 0 or is_first_steps or is_last_step)
  is_eval_step = valid_interval_steps and (step_ % valid_interval_steps == 0 or is_first_steps or is_last_step)

  if ops is not None:
    kwargs = {}   

    if FLAGS.profile_interval_steps and step_ % FLAGS.profile_interval_steps == 0:
      # https://blog.csdn.net/kenneth_yu/article/details/77466776
      if not hasattr(train_once, 'profiler'):
        train_once.profiler = model_analyzer.Profiler(graph=sess.graph)
        train_once.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        train_once.run_metadata = tf.compat.v1.RunMetadata()
        train_once.timeline_dir = f'{model_dir}/timeline'
        os.system(f'mkdir -p {train_once.timeline_dir}')
      kwargs['options'] = train_once.run_options 
      kwargs['run_metadata'] = train_once.run_metadata
                     
    feed_dict = {} if gen_feed_dict_fn is None else gen_feed_dict_fn()
    melt.feed_dict.update(feed_dict)
    with_summary = rank == 0 and log_dir and FLAGS.write_summary and train_once.summary_op is not None and (is_interval_step and not(eval_ops and is_eval_step))
    if not with_summary:
      results = sess.run(ops, feed_dict=melt.feed_dict, **kwargs) 
    else:
      results = sess.run(ops + [train_once.summary_op], feed_dict=melt.feed_dict, **kwargs)
      summary_str = results[-1]
      results = results[:-1]

    step += 1

    if FLAGS.profile_interval_steps and step % FLAGS.profile_interval_steps == 0:
      profile_step(train_once.profiler, step, train_once.run_metadata, train_once.timeline_dir)

    #reults[0] assume to be train_op, results[1] to be learning_rate
    learning_rate = results[1]
    results = results[2:]

    if print_avg_loss:
      if not hasattr(train_once, 'avg_loss'):
        train_once.avg_loss = AvgScore()
      loss = gezi.get_singles(results)
      gezi.set('loss', '%.4f' % loss[0])
      assert loss, 'No single result/op!'
      train_once.avg_loss.add(loss)
    
  if eval_names is None:
    if names:
      eval_names = ['' + x for x in names]
  
  if names:
    names = ['train/' + x for x in names]

  if eval_names:
    eval_names = ['' + x for x in eval_names]

  eval_str = ''
  # deal with summary
  if log_dir:
    if eval_ops and is_eval_step:
      for i in range(eval_loops):
        valid_feed_dict = {} if gen_eval_feed_dict_fn is None else gen_eval_feed_dict_fn()
        melt.valid_feed_dict.update(valid_feed_dict)
        with_summary = log_dir and train_once.summary_op is not None and FLAGS.write_summary and rank == 0 
        if not with_summary:
          eval_results = sess.run(eval_ops, feed_dict=melt.valid_feed_dict)
        else:
          eval_results = sess.run(eval_ops + [train_once.summary_op], feed_dict=melt.valid_feed_dict)
          summary_str = eval_results[-1]
          eval_results = eval_results[:-1]
        eval_loss = gezi.get_singles(eval_results)
        eval_stop = False
        if use_horovod:
          sess.run(hvd.allreduce(tf.constant(0)))

        eval_names_ = melt.adjust_names(eval_loss, eval_names)
        eval_str = 'valid:[{}]'.format(melt.parse_results(eval_loss, eval_names_))

        assert len(eval_loss) > 0
        if eval_stop is True:
          stop = True
        eval_names_ = melt.adjust_names(eval_loss, eval_names)
        if rank == 0:
          gezi.set('valid_loss', melt.parse_results(eval_loss, eval_names_))

  steps_per_second = None
  instances_per_second = None 
  hours_per_epoch = None

  # 当step是100 和 200 的时候会强制做验证展示(如果verbose模式),如果不想做验证设置--interval_steps=0 如果只想在100，200做两次验证 可以设置 --interval_steps=1e10（足够大的数）
  if is_interval_step and rank == 0:
    train_average_loss = train_once.avg_loss.avg_score()
    if print_time:
      duration = timer.elapsed()
      duration_str = 'duration:{:.2f} '.format(duration)
      melt.set_global('duration', '%.2f' % duration)
      elapsed = train_once.timer.elapsed()
      elapsed_steps = step - train_once.timer.step 
      train_once.timer.step = step
      steps_per_second = elapsed_steps / elapsed
      batch_size = melt.batch_size()
      num_gpus = int(melt.batch_size() / FLAGS.batch_size) if not gezi.is_cpu_only() else 0
      instances_per_second = elapsed_steps * batch_size / elapsed 
      instances_per_second_str = np.format_float_scientific(instances_per_second, precision=1, trim='0')
      gpu_info = ' gpus:[{}]'.format(num_gpus)
      if num_steps_per_epoch is None:
        epoch_time_info = ''
      else:
        try:
          hours_per_epoch = num_steps_per_epoch / elapsed_steps * elapsed / 3600
          mintues_per_epoch = hours_per_epoch * 60
          epoch_time_info = '1epoch:[{:.1f}h]'.format(hours_per_epoch) if hours_per_epoch > 1 else  '1epoch:[{:.1f}m]'.format(mintues_per_epoch)
          learning_rate_str = ','.join(np.format_float_scientific(x, precision=1, trim='0') for x in learning_rate)
          info.write('elapsed:[{:.2f}] batch_size:[{}]{} steps/s:[{:.1f}] insts/s:[{}] {} lr:[{}]'.format(
              elapsed, batch_size, gpu_info, steps_per_second, instances_per_second_str, epoch_time_info, learning_rate_str))
        except Exception:
          pass

    if print_avg_loss:
      names_ = melt.adjust_names(train_average_loss, names)
      info.write(' train:[{}] '.format(melt.parse_results(train_average_loss, names_)))
      gezi.set('loss', melt.parse_results(train_average_loss, names_))
    info.write(eval_str)
    logging.info2('{} {} {}'.format(epoch_str, 'step:%5d' % step, info.getvalue()))
    
    if deal_results_fn is not None:
      stop = deal_results_fn(results)

  metric_evaluate = False

  if metric_eval_fn is not None \
    and ((metric_eval_interval_steps \
         and step % metric_eval_interval_steps == 0) or model_path):
   metric_evaluate = True

  if 'QUICK' in os.environ:
    metric_evaluate = False

  if metric_evaluate:
    epoch_ = step / num_steps_per_epoch if num_steps_per_epoch else None
    FLAGS.train_time = timer_.elapsed_minutes() #train time not include metric evaluate 
    # logging.info(f'Round:{FLAGS.round}', 'Epoch:%.1f' % epoch_ , f'Train:{FLAGS.train_hour} Valid:{FLAGS.valid_hour}', 'TrainTime:{:.1f}m'.format(FLAGS.train_time))
    gstep = int(int(melt.epoch() * 10000) / int(valid_interval_epochs * 10000))
    if rank == 0:
      melt.inc_train_step()
      gstep = melt.inc_eval_step(save_file=(not FLAGS.async_valid))

    if not FLAGS.async_valid:
      evaluate_summaries = None
      if not model_path or 'model_path' not in inspect.getargspec(metric_eval_fn).args:
        metric_eval_fn_ = metric_eval_fn
      else:
        if FLAGS.write_valid_final and melt.epoch() == num_epochs:
          metric_eval_fn_ = lambda: metric_eval_fn(model_path=model_path, write=True)
        else:
          metric_eval_fn_ = lambda: metric_eval_fn(model_path=model_path)
      
      try:
        results = metric_eval_fn_()
        if isinstance(results, tuple):
          num_returns = len(results)    
          if num_returns == 2:
            evaluate_names, evaluate_vals = results
          else:
            assert num_returns == 3, 'retrun 1,2,3 ok 4.. not ok'
            evaluate_names, evaluate_vals, evaluate_summaries = results
        else:  # return dict
          evaluate_names, evaluate_vals = list(zip(*results.items()))   
      except Exception:
        logging.info('Do nothing for metric eval fn with exception:\n', traceback.format_exc())
        evaluate_names = []
        evaluate_vals = []

      if rank == 0:
        if evaluate_names:
          names = [x.replace('eval/', '') for x in evaluate_names]
          gezi.set('metric_names', list(names))
          gezi.set('metric_values', list(evaluate_vals)) 
          gezi.set('result', dict(zip(names, evaluate_vals)))
          if len(evaluate_names) > 15:
            if FLAGS.version not in ['train', 'valid', 'test']:
              logging.info('{} valid_step:{} valid_metrics:{}'.format(epoch_str, step, ['%s:%.4f' % (name, val) for name, val in zip(names[:30], evaluate_vals[:30]) if not isinstance(val, str)] + ['version:{}'.format(FLAGS.version)]))
            else:
              logging.info('{} valid_step:{} valid_metrics:{}'.format(epoch_str, step, ['%s:%.4f' % (name, val) for name, val in zip(names[:30], evaluate_vals[:30]) if not isinstance(val, str)]))
          else:
            results = dict(zip(names, evaluate_vals))
            gezi.pprint_dict(gezi.dict_rename(results, 'Metrics/', ''), print_fn=logging.info, 
                             desc='{} valid_step:{} version:{} valid_metrics:'.format(epoch_str, step, FLAGS.version),
                             format='%.4f')

      if learning_rate is not None and (learning_rate_patience and learning_rate_patience > 0):
        assert learning_rate_decay_factor > 0 and learning_rate_decay_factor < 1
        valid_loss = evaluate_vals[0]
        if not hasattr(train_once, 'min_valid_loss'):
          train_once.min_valid_loss = valid_loss
          train_once.deacy_steps = []
          train_once.patience = 0
        else:
          if valid_loss < train_once.min_valid_loss:
            train_once.min_valid_loss = valid_loss
            train_once.patience = 0
          else:
            train_once.patience += 1
            logging.info2('{} valid_step:{} patience:{}'.format(epoch_str, step, train_once.patience))
        
        if learning_rate_patience and train_once.patience >= learning_rate_patience:
          lr_op = ops[1]
          lr = sess.run(lr_op) * learning_rate_decay_factor
          train_once.deacy_steps.append(step)
          logging.info2('{} valid_step:{} learning_rate_decay by *{}, learning_rate_decay_steps={}'.format(epoch_str, step, learning_rate_decay_factor, ','.join(map(str, train_once.deacy_steps))))
          sess.run(tf.compat.v1.assign(lr_op, tf.constant(lr, dtype=tf.float32)))
          train_once.patience = 0
          train_once.min_valid_loss = valid_loss
    else:
      return -1
  
  summary_strs = gezi.to_list(summary_str)  
  if metric_evaluate:
    if evaluate_summaries is not None:
      summary_strs += evaluate_summaries 

  # for horovod hvd.rank != 0 log_dir is already set as None, so log_dir==None means rank==0
  if log_dir:
    if is_eval_step and not is_start:
      total_step = melt.get_total_step()
      step_ = step + total_step
      # deal with summary
      if FLAGS.write_summary:
        summary_writer = melt.get_summary_writer()
        summary = tf.compat.v1.Summary()
        if eval_ops is None:
          if train_once.summary_op is not None:
            for summary_str in summary_strs:
              train_once.summary_writer.add_summary(summary_str, step_)
        else:
          for summary_str in summary_strs:
            train_once.summary_writer.add_summary(summary_str, step_)
          suffix = 'valid' if not eval_names else ''
          for val, name in zip(eval_loss, eval_names_):
            summary_writer.scalar(f'{name}/valid', val, step_, 0)
          if train_once.valid_writer:
            for val, name in zip(eval_loss, eval_names_):
              train_once.valid_writer.scalar(name, val, step_, 0)

        if ops is not None:
          for val, name in zip(train_average_loss, names_):
            summary_writer.scalar(f'{name}/train', val, step_, 0)
          if train_once.train_writer:
            for val, name in zip(train_average_loss, names_):
              train_once.train_writer.scalar(name, val, step_)
      
          for i in range(len(learning_rate)):
            name = 'learning_rate' if i == 0 else 'learning_rate_%d' % i
            summary_writer.scalar(name, learning_rate[i], step_, 0)
          summary_writer.scalar('other/batch_size', melt.batch_size(), step_, 0)
          summary_writer.scalar('other/epoch', melt.epoch(), step_, 0)
          summary_writer.scalar('other/round', FLAGS.round, step_, 0)
          #if FLAGS.train_hour:
          #  summary_writer.scalar('other/train_time', int(FLAGS.train_hour), step_, 0)
          if steps_per_second:
            summary_writer.scalar('perf/steps_per_second', steps_per_second, step_, 0)
          if instances_per_second:
            summary_writer.scalar('perf/instances_per_second', instances_per_second, step_, 0)
          if hours_per_epoch:
            summary_writer.scalar('perf/hours_per_epoch', hours_per_epoch, step_, 0)
          summary_writer.scalar('other/step', step_, step_, 0)
          summary_writer.scalar('other/step2', step_, step_)
      
        train_once.summary_writer.add_summary(summary, step_)
        train_once.summary_writer.flush()
      
    if metric_evaluate:
      prefix = 'step_eval'
      if model_path:
        prefix = ''
      if prefix:
        evaluate_names = [f'{prefix}/{name}' for name in evaluate_names]

      if not FLAGS.async_valid:
        melt.write_metric_summaries(evaluate_names, evaluate_vals, gstep)

    return stop
  
