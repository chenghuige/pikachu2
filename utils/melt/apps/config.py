#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2019-09-07 10:59:47.160833
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

# from addict import Dict

# new tf seems to try to use absl which has def of log_dir TODO
try:
  flags.DEFINE_string('log_dir', None, '''if none will set to model_dir
                                        ''')
except Exception:
  pass

flags.DEFINE_string('ori_log_dir', None, '')
flags.DEFINE_bool('log_dir_under_model_dir', False, '')
# download fro gcs
flags.DEFINE_string('gcs_root', None, 'this is work with wandb_id')
flags.DEFINE_alias('gcs_dir', 'gcs_root')
flags.DEFINE_string('gcs_src', None, '')
flags.DEFINE_string('gcs_dest', None, '')
flags.DEFINE_bool('gcs_sync', False, '')

#-------input data
flags.DEFINE_integer('base_batch_size', None, '')
flags.DEFINE_integer('batch_size', 32, 'Batch size. default as im2text default')
flags.DEFINE_alias('bs', 'batch_size')
flags.DEFINE_integer('eval_batch_size', None, 'Batch size fore eval')
flags.DEFINE_alias('ebs', 'eval_batch_size')
flags.DEFINE_float('valid_multiplier', 1., '')
flags.DEFINE_alias('valid_mul', 'valid_multiplier')
flags.DEFINE_alias('eval_mul', 'valid_multiplier')
flags.DEFINE_bool('drop_remainder', None, 'then by default train True, other False, but for tpu all True')
flags.DEFINE_integer('prefetch', None, '')

flags.DEFINE_integer('global_batch_size', None, '')
flags.DEFINE_integer('replica_batch_size', None, '')
flags.DEFINE_integer('global_eval_batch_size', None, '')
flags.DEFINE_integer('replica_eval_batch_size', None, '')

flags.DEFINE_integer('gradient_accumulation_steps', 1, '')
flags.DEFINE_alias('acc_steps', 'gradient_accumulation_steps')

#-------flow
flags.DEFINE_float('num_epochs', 1, '''Number of epochs to run trainer.
                                         0 means run forever epochs,
                                         -1 mens you want to just run 1 step and stop!, usefull when you want to do eval all once''')
flags.DEFINE_alias('epochs', 'num_epochs')
flags.DEFINE_alias('ep', 'num_epochs')
flags.DEFINE_integer('num_epochs2', None, '')
flags.DEFINE_integer('num_steps', 0, '''Number of steps to run trainer. 0 means run forever, 
                                        -1 means you just want to build graph and save without training(changing model value)''')
flags.DEFINE_alias('steps', 'num_steps')
#-------model
flags.DEFINE_string('model', '', '')
# flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_alias('model_dir', 'log_dir')
flags.DEFINE_string('datadir', None, '')
flags.DEFINE_bool('check_loaded', False, '')
flags.DEFINE_string('pretrained_dir', None, 'if set is read only pretrained dir mainly for fintune different from model_dir(save)')
flags.DEFINE_alias('pretrain', 'pretrained_dir')
flags.DEFINE_alias('pretrained', 'pretrained_dir')
flags.DEFINE_alias('pt', 'pretrained_dir')
flags.DEFINE_string('model_path', None, 'input model path to reload if not None otherwise try latest checkpoint from model_dir')
flags.DEFINE_string('del_model_path', None, '')
flags.DEFINE_string('ori_model_dir', None, '')
flags.DEFINE_string('model_root', None, '')
flags.DEFINE_string('model_suffix', None, '')
flags.DEFINE_bool('model_time_suffix', False, '')
flags.DEFINE_alias('mts', 'model_time_suffix')
flags.DEFINE_string('model_name_suffix', None, '')
flags.DEFINE_alias('mns', 'model_name_suffix')
flags.DEFINE_bool('model_seed_suffix', False, '')
flags.DEFINE_alias('mss', 'model_seed_suffix')
flags.DEFINE_bool('disable_model_suffix', False, '')
flags.DEFINE_string('model_name', None, '')
flags.DEFINE_alias('mname', 'model_name')
flags.DEFINE_alias('mn', 'model_name')
flags.DEFINE_boolean('save_model', True, '')
flags.DEFINE_alias('save', 'save_model')
flags.DEFINE_boolean('save_only', False, 'only start init and save model might be usefull to check model init vars')
flags.DEFINE_boolean('save_graph', False, 'some model might not save any weights, and no error produce which is annoying, TODO FIXME, so just by default model.save_weights')
flags.DEFINE_alias('save_network', 'save_graph')
flags.DEFINE_boolean('save_graph_slim', False, '')
flags.DEFINE_boolean('save_weights', False, 'normaly do not need as we have saved model.h5 with graph and weights if fail will save model.h5 with weights only, so here True means model.h5 with graph and dum weights.h5 with weight only')
flags.DEFINE_boolean('save_inter_models', False, 'by default False not save model.h5 only checkpoint if not is last')
flags.DEFINE_boolean('load_graph', True, '')
flags.DEFINE_boolean('load_weights_only', False, '')
flags.DEFINE_alias('lwo', 'load_weights_only')
flags.DEFINE_boolean('load_by_name', True, '')
flags.DEFINE_boolean('load_skip_mismatch', True, '')
flags.DEFINE_boolean('restore_configs', False, '')
flags.DEFINE_boolean('save_optimizer', True, '')
flags.DEFINE_boolean('saved_model', False, '')
flags.DEFINE_boolean('save_checkpoint', True, '')
flags.DEFINE_boolean('save_keras_model', True, '')
flags.DEFINE_boolean('model2tb', False, '')
flags.DEFINE_float('save_interval_epochs', 1, 'if -1 will not save, by default 1 epoch 1 model in modeldir/epoch, you can change to 2, 0.1 etc')
flags.DEFINE_alias('sie', 'save_interval_epochs')
flags.DEFINE_float('save_interval_seconds', 0, 'model/checkpoint save interval by n seconds, if > 0 will use this other wise use save_interval_hours')
flags.DEFINE_float('save_interval_hours', 10000, """model/checkpoint save interval by n hours""")
flags.DEFINE_float('save_interval_steps', 100000000, 'model/checkpoint save interval steps')
flags.DEFINE_boolean('save_epoch_ckpt', False, '')
flags.DEFINE_bool('freeze_graph', False, '''if True like image caption set this to True, will load model faster when inference,
                                            but sometimes it is slow to freeze, might due to tf bug, then you can set it to False,
                                            if you do not need inference also set to False to speed up training,
                                            TODO FIXME for cudnn right now if not set feeze graph can not load, is a bug of melt ?''')
flags.DEFINE_bool('freeze_graph_final', True, '')
flags.DEFINE_integer('max_models_keep', 1, 'keep recent n models, default 2 for safe')
flags.DEFINE_boolean('restore_from_latest', True, 'more safe to restore from recent but not latest')
flags.DEFINE_boolean('restore_optimizer', True, '')
flags.DEFINE_boolean('del_inter_model', False, '')
flags.DEFINE_alias('del', 'del_inter_model')
flags.DEFINE_boolean('del_inter_events', False, '')

#--------show
flags.DEFINE_integer('interval_steps', 100, '')
flags.DEFINE_alias('is', 'interval_steps')
flags.DEFINE_integer('valid_interval_steps', 0,   """for training suggest 100 if you want to see valid loss, and other valid metrics if set during training, 
                                                     you can check evaluate interval time and evaluate once time the ratio below 0.1,
                                                     However due to bug prone with tf new version, disable it by default to 0
                                                     """)                                                     
flags.DEFINE_alias('vis', 'valid_interval_steps')
flags.DEFINE_integer('metric_eval_interval_steps', 0, 'if > 0 need to be valid_interval_steps * n')
flags.DEFINE_boolean('metric_eval', True, '')
flags.DEFINE_list('metrics', [], '')
flags.DEFINE_list('test_names', [], '')

flags.DEFINE_float('train_interval_epochs', 1, '')
flags.DEFINE_float('valid_interval_epochs', 1, 'if 0 will be the same as num_epochs so only valid at the end, if > num_epochs will be num_epochs, if -1 will disable valid interval epochs')
flags.DEFINE_alias('vie', 'valid_interval_epochs')
flags.DEFINE_integer('num_eval_steps', None, '')
flags.DEFINE_integer('eval_verbose', 1, '')
flags.DEFINE_alias('num_valid_steps', 'num_eval_steps')
flags.DEFINE_alias('nvs', 'num_eval_steps')
flags.DEFINE_alias('nes', 'num_eval_steps')
flags.DEFINE_bool('do_valid', True, '')
flags.DEFINE_bool('do_valid_last', True, '')
flags.DEFINE_integer('num_loop_dirs', None, '')
flags.DEFINE_string('valid_exclude', None, '')
flags.DEFINE_string('train_exclude', None, '')
flags.DEFINE_bool('allow_valid_train', False, '')
flags.DEFINE_alias('allow_train_valid', 'allow_valid_train')
flags.DEFINE_float('first_interval_epoch', -1., 'default -1 means not to use, if set like 0.1 then will do first epoch validation on 0.1epoch')
flags.DEFINE_float('second_interval_epoch', -1., 'default -1 means not to use, if set like 0.1 then will do first epoch validation on 0.1epoch')
flags.DEFINE_float('test_interval_epochs', 0, 'same meaning as valid_interval_epochs, by default 0 means only test at train end')
flags.DEFINE_alias('tie', 'test_interval_epochs')
flags.DEFINE_bool('do_test', True, '')
flags.DEFINE_alias('dt', 'do_test')
flags.DEFINE_float('inference_interval_epochs', 1, '')
flags.DEFINE_boolean('write_during_train', True, '')

flags.DEFINE_integer('eval_loops', 1, 'set to max inorder to hack for evaluation..')
flags.DEFINE_bool('custom_eval_loop', False, '')
flags.DEFINE_bool('custom_test_loop', False, '')

#----------optimize
#flags.DEFINE_string('optimizer', 'adadelta', 'follow squad of ukhst https://www.quora.com/Why-is-AdaDelta-not-favored-in-Deep-Learning-communities-while-AdaGrad-is-preferred-by-many-over-other-SGD-variants')
flags.DEFINE_string('optimizer', 'adam', 'follow squad of ukhst https://www.quora.com/Why-is-AdaDelta-not-favored-in-Deep-Learning-communities-while-AdaGrad-is-preferred-by-many-over-other-SGD-variants')
flags.DEFINE_alias('opt', 'optimizer')
flags.DEFINE_string('optimizer2', None, '')
flags.DEFINE_alias('opt2', 'optimizer2')
flags.DEFINE_integer('num_optimizers', None, '')
flags.DEFINE_integer('num_learning_rates', None, '')
flags.DEFINE_float('opt_epsilon', 1e-6, 'follow squad of ukhst/bert also seems adam default 1e-8 will cause l2 too large and easy to overfit')
flags.DEFINE_string('optimizers', None, '')
flags.DEFINE_alias('opts', 'optimizers')
flags.DEFINE_float('opt_weight_decay', 0., 'or 0.01 for safe which will have less l2 loss')
flags.DEFINE_bool('opt_amsgrad', False, 'for torch adam')
flags.DEFINE_float('opt_momentum', 0.9, 'for torch sgd')
flags.DEFINE_bool('opt_nesterov', False, '')
flags.DEFINE_bool('opt_ema', False, '')
flags.DEFINE_float('opt_ema_momentum', 0.9999, '')
flags.DEFINE_bool('opt_swa', False, '')

flags.DEFINE_bool('ema_inject', False, '')

flags.DEFINE_integer('opt_accumulate_steps', 1, '')
flags.DEFINE_alias('opt_acc_steps', 'opt_accumulate_steps')

flags.DEFINE_bool('multiopt_train_step', False, 'if False use tfa.optimizers.MultiOptimizer otherwise use melt.util.multiopt_train_step')

flags.DEFINE_float('l2_weight_decay', 0., '')
flags.DEFINE_float('learning_rate', 0.001, """adam will be 0.001
                                         default is adam default lr""")
flags.DEFINE_alias('lr', 'learning_rate')
flags.DEFINE_float('learning_rate2', 0.001, """adam will be 0.001
                                         default is adam default lr""")
flags.DEFINE_alias('lr2', 'learning_rate2')
flags.DEFINE_list('learning_rates', [], """adam will be 0.001
                                         default is adam default lr""")
flags.DEFINE_alias('lrs', 'learning_rates')                                      
flags.DEFINE_float('min_learning_rate', 0., 'min learning rate used for dyanmic eval metric decay')
flags.DEFINE_alias('min_lr', 'min_learning_rate')
flags.DEFINE_float('learning_rate_start_factor', 1., '')

flags.DEFINE_float('learning_rate_decay_power', 1., '0.5 will be slower decrease, 1.5 will be faster')
flags.DEFINE_alias('lr_decay_power', 'learning_rate_decay_power')
flags.DEFINE_string('learning_rate_decay_method', 'poly', '')
flags.DEFINE_alias('lr_decay_method', 'learning_rate_decay_method')

flags.DEFINE_bool('learning_rate_scale_bygpu', False, '')
flags.DEFINE_alias('lr_scale_bygpu', 'learning_rate_scale_bygpu')

flags.DEFINE_float('learning_rate_multiplier', None, '')
flags.DEFINE_alias('lr_mul', 'learning_rate_multiplier')
flags.DEFINE_alias('lr_scale', 'learning_rate_multiplier')

flags.DEFINE_float('loss_scale', 1., '')
flags.DEFINE_alias('loss_mul', 'loss_scale')

flags.DEFINE_bool('reset_all', False, '')
flags.DEFINE_bool('reset_learning_rate', False, '')
flags.DEFINE_alias('reset_lr', 'reset_learning_rate')
flags.DEFINE_bool('reset_global_step', False, '')

#flags.DEFINE_float('learning_rate_decay_factor', 0.97, 'im2txt 0.5, follow nasnet using 0.97')
flags.DEFINE_float('learning_rate_decay_factor', 0., 'im2txt 0.5, follow nasnet using 0.97')
flags.DEFINE_alias('lr_decay', 'learning_rate_decay_factor')
flags.DEFINE_boolean('dynamic_learning_rate', False, '')
flags.DEFINE_alias('dynamic_lr', 'dynamic_learning_rate')
flags.DEFINE_integer('learning_rate_patience', None, 'might be 3 for 3 times eval loss no decay')
flags.DEFINE_alias('lr_patience', 'learning_rate_patience')

flags.DEFINE_float('decay_start_epoch', 0., 'start decay from epoch')
flags.DEFINE_float('num_epochs_per_decay', 2.4, 'follow nasnet')
flags.DEFINE_integer('decay_start_step', 0, 'start decay from steps')
flags.DEFINE_integer('num_steps_per_decay', 0, 'if 0 no effect, if > 0 then will not use this instead of num_epochs_per_decay')
flags.DEFINE_string('learning_rate_values', None, 'like 0.1,0.05,0.005')
flags.DEFINE_alias('lr_values', 'learning_rate_values')
flags.DEFINE_string('learning_rate_step_boundaries', None, 'like 10000,20000')
flags.DEFINE_string('learning_rate_epoch_boundaries', None, 'like 10,30 or 10.5,30.6')
flags.DEFINE_alias('lr_epoch_boundaries', 'learning_rate_epoch_boundaries')
flags.DEFINE_integer('num_learning_rate_weights', 0, '')
flags.DEFINE_float('warmup_proportion', 0.1, '0.1 might be a good choice')
flags.DEFINE_alias('warmup_ratio', 'warmup_proportion')
flags.DEFINE_float('warmup_epochs', 0, '')
flags.DEFINE_integer('sustain_epochs', 0, '')
flags.DEFINE_float('epoch_lr_exp_decay', 1., '')
flags.DEFINE_integer('epoch_lr_decay_strategy', 1, '策略0是kaggle tpu flower给出的每次*0.8类似这样 策略1仿照bert三角学习率 只是每次轮中间学习率保持不变')
flags.DEFINE_integer('warmup_steps', 0, 'if set then warmup proportion not on effect, can be set like 2000')
flags.DEFINE_float('num_decay_epochs', None, '')
flags.DEFINE_alias('decay_epochs', 'num_decay_epochs')

flags.DEFINE_float('swa_start_epoch', None, '1,2... 1 means the second epoch(epoch start from 0)')
flags.DEFINE_alias('swa_start', 'swa_start_epoch')
flags.DEFINE_float('swa_lr_ratio', 0.1, '')
flags.DEFINE_float('swa_freq', 1, '')
flags.DEFINE_float('swa_warmup', 0., '')
flags.DEFINE_bool('swa_fake_run', False, '')

flags.DEFINE_integer('sart_epoch', 0, '')

#cosine learning rate method
flags.DEFINE_float('learning_rate_cosine_t_mul', 2., '')
flags.DEFINE_float('learning_rate_cosine_m_mul', 1., '')
flags.DEFINE_float('learning_rate_cosine_alpha', 0., '')

flags.DEFINE_string('learning_rate_method', 'none', 'decay or cosine or none')
flags.DEFINE_alias('lr_method', 'learning_rate_method')

flags.DEFINE_bool('fuxian_old_lr', False, '')
flags.DEFINE_bool('bert_style_lr', True, 'if set bert_style_lr==0 will change bert-adam to adam')

#flags.DEFINE_string('lr_ratios', None, '0.2,1,0.2,1,1,1')
flags.DEFINE_boolean('use_finetune_step', False, '')
flags.DEFINE_boolean('use_finetune2_step', False, '')

flags.DEFINE_float('clip_gradients', 5.0, """follow im2text as 5.0 default, 
                                          set to 1.0 in deeipiu/image_caption try sh ./train/flickr-rnn.sh, 
                                          will show perf from 900inst/s to 870ints/s and also slow convergence""")
flags.DEFINE_boolean('optimize_has_scope', True, '')

#----------train
flags.DEFINE_boolean('train_only', False, '')
flags.DEFINE_boolean('train_all', False, 'use for small dataset like competetion or standard dataset where use k folds for train/valid and use all k parts if set train_all==True')
flags.DEFINE_boolean('train_scratch', False, '')
flags.DEFINE_boolean('clear_model', False, '')
flags.DEFINE_boolean('clear', False, '')
flags.DEFINE_boolean('clear_first', False, '')
flags.DEFINE_alias('cf', 'clear_first')
flags.DEFINE_string('work_mode', 'train', '')
flags.DEFINE_string('mode', None, '')
flags.DEFINE_integer('monitor_level', 2, '1 will monitor emb, 2 will monitor gradient')
flags.DEFINE_boolean('disable_monitor', False, '')
flags.DEFINE_integer('log_level', 20, '20 is logging.INFO, 10 is logging.DEBUG')
flags.DEFINE_boolean('no_log', False, '')
flags.DEFINE_boolean('quiet', True, 'do not log per step info to screen but still in logfile')
flags.DEFINE_boolean('verbose', None, '')
#flags.DEFINE_string('mode', 'train', 'or predict')
flags.DEFINE_boolean('freeze_graph_collections', True, '')

flags.DEFINE_boolean('use_tower_loss', False, 'prefer to use horovod')
flags.DEFINE_bool('use_valid_loss', True, '')

flags.DEFINE_boolean('recount_tfrecords', False, '')
flags.DEFINE_boolean('recount_train_steps', False, '')
flags.DEFINE_integer('min_tfrecords', 10000, '')
flags.DEFINE_bool('static_input', False, '')
flags.DEFINE_alias('record_padded', 'static_input')
flags.DEFINE_bool('fixed_pad', False, '')
flags.DEFINE_bool('dynamic_pad', False, '')
flags.DEFINE_alias('padded_batch', 'dynamic_pad')
flags.DEFINE_bool('save_eager_ckpt', False, '')

#----------multi gpu
##TODO be carefull to use mult gpu mode, since for some case it will casue performance lower then single gpu mode 
##especially for finetune image model or some models that will cause sess.run summary catch exception FIXME
##also google paper lessons from 2015 coco caption contest show and tell model will have low performance also using 
##multi gpu so they use single gpu training
flags.DEFINE_integer('num_replicas', 1, '')
flags.DEFINE_integer('num_gpus', None, """How many GPUs to use. set 0 or None to disable multi gpu mode, set like 1024 or -1 to use all available gpus""")
flags.DEFINE_alias('gpus', 'num_gpus')
flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
flags.DEFINE_boolean('batch_size_per_gpu', False, '''  per gpu batch size should be dived by gpu num ?
                                                      True means, if num_gpus = 2, batch_size set 128, then each gpu with batch size 128, which means batch_size is actually batch_size_per_gpu
                                                      means 256 insts per step actually, if num_gpus == 0, will try to read env info if you set like 
                                                      CUDA_VISIABLE_DIVICES=0,1 then actually will use 2 GPUS, and 256 insts per step also 
                                                      if not find CUDA_VISIABLE_DIVICES infoep then, just use 1 GPU, single gpu mode, 128 insts per step
                                                      if num_gpus == 1, also 1GPU, 128 insts per step, but will use tower_loss with 1 gpu(mainly for test if tower_loss ok)
                                                      not used much, so if you want use single gpu, just set num_gpus=0 
                                                      if batch_size_per_gpu = False, with 2gpu, then it means each GPU will be of batch size 128 / 2 = 64, total insts 
                                                      are still 128 per step
                                                      batch_size_per_gpu False is better for debug, all program, deal same num instances so better for comparation
                                                      batch_size_per_gpu True is better for speed up, fully use multi gpu, like one gpu can only train batch 32(OOM for
                                                      bigger batch_size) then 2 gpu can deal 2 * 32 instances per step
                                                      For experiment simplicity, set it to True by default, same instances per step if increase gpu num
                                                     ''') 
flags.DEFINE_alias('bspg', 'batch_size_per_gpu')                                                     
flags.DEFINE_string('variable_strategy', 'cpu', '')
flags.DEFINE_integer('min_free_gpu_mem', 0, 'by M, if 0 or < 0 means need all gpu mem exclusive gpu mode')
flags.DEFINE_integer('min_free_gpu_mem_valid', 0, 'by M, if 0 or < 0 means need all gpu mem exclusive gpu mode')
flags.DEFINE_integer('max_used_gpu_mem', None, 'by M 20M means exclusive only 1 process 1 gpu but for some machine it might cosume like 200M by default like p40, so for tf set allow_growth=False maybe better')
flags.DEFINE_string('hack_device', None, 'set it to cpu if you want to put some part on cpu when gpu mem is low')
flags.DEFINE_bool('allow_cpu', True, 'for async_valid wether allow to restart to use cpu for valid')

#----------scope
flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                      set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')
flags.DEFINE_string('restore_include', '', '')
flags.DEFINE_string('restore_exclude', '', '')
flags.DEFINE_string('restore_include2', '', '')
flags.DEFINE_string('restore_exclude2', '', '')
flags.DEFINE_string('main_scope', 'main', 'or use other main_scope like run, this is mainly graph scope for varaible reuse')

#----- cross fold
flags.DEFINE_integer('fold', None, '')
flags.DEFINE_integer('num_folds', None, 'if 5 means 5 fold cv')
flags.DEFINE_alias('folds', 'num_folds')
flags.DEFINE_integer('start_fold', None, '')
flags.DEFINE_alias('sfold', 'start_fold')
flags.DEFINE_bool('cv_save_weights', False, '')
flags.DEFINE_bool('cv_save_temp', False, '')
flags.DEFINE_bool('cv_valid_only', False, 'no train just load models and evaluate')
flags.DEFINE_alias('cvo', 'cv_valid_only')
flags.DEFINE_bool('cv_test_only', False, 'no train just load models and evaluate')
flags.DEFINE_alias('cto', 'cv_test_only')
flags.DEFINE_string('train_mark', 'train', '')
flags.DEFINE_string('valid_mark', 'valid', '')
flags.DEFINE_string('test_mark', 'test', '')

flags.DEFINE_bool('fp16', False, 'tpu可以优先fp16模式速度更快 但是很奇怪比float32更占存储空间更容易OOM 所以如果OOM优先先尝试fp16=False 如果还是OOM再batch_size减半')
flags.DEFINE_alias('float16', 'fp16')
flags.DEFINE_bool('dataset_fp16', True, 'only work with when fp16 == True')

# multi parts infer
# 为了区别part 自定义处理不走dataset shard 这部分有点乱 另外包括 task_index ps world_size rank 
# 这里是要走dataset shard主要好处是 可能数据集tfrecord比如只有一个文件这种 依然能够切分数据并行  
# 走dataset shard 可以配合 --shard_by_files只对文件shard减少数据读取 如果能整除并行的数目没问题 如果不能 那么是 dataset.shard不是让最后一个worker少分担 而是多分担 效率会变低 7min (40文件 6gpu dataset shard 5*6 + 1* 10)
# 不能整除不如自己分割file 让最后一个worker少分担一点  (40文件 6gpu 5*7 + 1* 5)  尽量还是数据考虑并行数目 能文件均分 数目均分比较好 6min 24 不过如果数据读取不是太大瓶颈 如果文件不能整除 不使用shard_by_files 6min 02
# 如果40文件 5gpu
# 自行切分文件  6min 59
# dataset shard by files 6min 52
# dataset shard 07:02
# 尽量均匀划分 能均分整除 使用dataset 如果不能均分 如果数据读取不是瓶颈 仍然用dataset 如果是瓶颈使用partition自己处理 这个逻辑也封装到 eval callback和 test callback
flags.DEFINE_integer('part', None, '')
flags.DEFINE_integer('num_parts', None, '')
flags.DEFINE_alias('parts', 'num_parts')
flags.DEFINE_bool('use_shard', False, '默认use shard就是走dataset shard,如果走evalcall back和test callback自行处理设置False')

#---------- input reader
flags.DEFINE_boolean('feed_dataset', True, 'for non eager mode')
  
flags.DEFINE_list('buckets', [], 'empty meaning not use, other wise looks like 5,10,15,30')
flags.DEFINE_list('batch_sizes', [], '')
flags.DEFINE_float('batch_size_scale', 1., '')
flags.DEFINE_alias('bs_scale', 'batch_size_scale')
flags.DEFINE_alias('bs_mul', 'batch_size_scale')
flags.DEFINE_integer('length_index', 1, '')
flags.DEFINE_string('length_key', None, '')
flags.DEFINE_bool('batch_parse', True, 'if batch_parse first batch then map')

flags.DEFINE_boolean('sparse_to_dense', True, '')
flags.DEFINE_boolean('padded_tfrecord', False, '')
flags.DEFINE_integer('padding_idx', 0, '')
flags.DEFINE_boolean('hvd_sparse_as_dense', False, 'horovd see lays/otimipzer.py hvd.DistributedOptimizer init args')
flags.DEFINE_string('hvd_device_dense', '/cpu:0', 'horovd see lays/otimipzer.py hvd.DistributedOptimizer init args, may set to /cpu:0')

flags.DEFINE_integer('min_after_dequeue', 0, """by deafualt will be 500, 
                                                set to large number for production training 
                                                for better randomness""")
flags.DEFINE_integer('buffer_size', 1024, """set to large number for production training 
                                             for better randomness""")
flags.DEFINE_integer('num_prefetch_batches', 1024, '')
flags.DEFINE_bool('repeat_then_shuffle', False, 'repeat在shuffle之前使用能提高性能，但模糊了数据样本的epoch关系')

#---------- input dirs
#@TODO will not use input pattern but use dir since hdfs now can not support glob well
flags.DEFINE_string('train_input', None, 'must provide')
flags.DEFINE_list('train_files', [], '')
flags.DEFINE_alias('input_files', 'train_files')
flags.DEFINE_alias('input', 'train_input')
flags.DEFINE_alias('inputs', 'input_files')
flags.DEFINE_integer('num_train', None, '')
flags.DEFINE_integer('min_train', 0, '')
flags.DEFINE_string('train_input2', None, 'addtional train input')
flags.DEFINE_alias('input2', 'train_input2')
flags.DEFINE_string('valid_input', None, 'if empty will train only')
flags.DEFINE_list('valid_files', [], '')
flags.DEFINE_alias('valid_inputs', 'valid_files')
flags.DEFINE_string('valid_out_file', 'valid.csv', '')
flags.DEFINE_integer('num_valid', None, '')
flags.DEFINE_integer('num_full_valid', None, '')
flags.DEFINE_integer('min_valid', 0, '')
flags.DEFINE_string('valid_input2', None, 'some applications might need another valid input')
flags.DEFINE_string('test_input', None, 'maily for inference during train epochs')
flags.DEFINE_list('test_files', None, '')
flags.DEFINE_string('test_out_file', 'submission.csv', '')
flags.DEFINE_bool('infer_write_streaming', False, '')
flags.DEFINE_integer('num_test', None, '')
flags.DEFINE_string('fixed_valid_input', None, 'if empty wil  not eval fixed images')
flags.DEFINE_string('num_records_file', None, '')
flags.DEFINE_string('base_dir', '../../../mount', '')
flags.DEFINE_boolean('cache', False, '')
flags.DEFINE_boolean('cache_train', False, '')
flags.DEFINE_boolean('cache_valid', False, '')
flags.DEFINE_boolean('cache_valid_input', False, '')
flags.DEFINE_boolean('cache_test', False, '')
flags.DEFINE_boolean('cache_test_input', False, '')
flags.DEFINE_string('cache_file', '', '')

flags.DEFINE_boolean('full_validation_final', False, '')
flags.DEFINE_alias('full_valid_last', 'full_validation_final')
flags.DEFINE_alias('full_valid_final', 'full_validation_final')
flags.DEFINE_alias('full_validation_last', 'full_validation_final')

flags.DEFINE_float("train_time", None, "time spent on training one run(round)")
flags.DEFINE_float("valid_time", None, "")
flags.DEFINE_float("total_time", None, "")

flags.DEFINE_boolean('run_valid_op', True, '')
flags.DEFINE_boolean('show_eval', True, '')
flags.DEFINE_boolean('eval_shuffle_files', False, '')
flags.DEFINE_boolean('shuffle', True, 'shuffle means shuffle batch and train files')
flags.DEFINE_boolean('shuffle_valid', False, 'shuffle means shuffle batch and valid files')
flags.DEFINE_boolean('shuffle_batch', None, 'shuffle batch only')
flags.DEFINE_boolean('shuffle_files', None, 'shuffle files only')
flags.DEFINE_boolean('parallel_read_files', True, '')
flags.DEFINE_boolean('fixed_random', False, '')
flags.DEFINE_integer('eval_seed', 2048, '')
flags.DEFINE_integer('seed', None, '1024 input seed')
flags.DEFINE_boolean('fix_sequence', False, '')

flags.DEFINE_string('big_buckets', None, 'empty meaning not use, other wise looks like 5,10,15,30')
flags.DEFINE_string('big_batch_sizes', None, '')
flags.DEFINE_integer('big_batch_size', None, '')

flags.DEFINE_boolean('adjust_global_step', False, '')
flags.DEFINE_integer('global_step', None, '')
flags.DEFINE_integer('global_epoch', None, '')

flags.DEFINE_boolean('eager', False, 'fo tf1 setting to eager, as tf1 default is graph mode')
flags.DEFINE_boolean('graph', False, 'for tf2 setting to graph, as tf2 default is eager mode, TODO now in melt.flow.flow.py using slim which not ok for tf2')
flags.DEFINE_boolean('run_eagerly', False, 'for keras model.compile')
flags.DEFINE_alias('eagerly', 'run_eagerly')

flags.DEFINE_integer('num_threads', 0, """threads for reading input tfrecords,
                                           setting to 1 may be faster but less randomness
                                        """)

flags.DEFINE_boolean('torch', False, 'torch_model Wether use torch model, if true and not torch_only, tf eager read, torch model')
flags.DEFINE_boolean('torch_only', False, 'torch_read Wether use torch reader, if true, torch read, torch model')
flags.DEFINE_boolean('torch_lr', True, 'False might not work as tf learning rate problem')
flags.DEFINE_boolean('torch_restart', False, '')

flags.DEFINE_boolean("keras", None, "using keras fit")
flags.DEFINE_boolean('keras_validation', False, '')
flags.DEFINE_integer('keras_validation_steps', 0, '')
flags.DEFINE_alias('keras_valid', 'keras_validation')
flags.DEFINE_alias('keras_eval', 'keras_validation')
flags.DEFINE_alias('kvalid', 'keras_validation')
flags.DEFINE_alias('keval', 'keras_validation')
flags.DEFINE_boolean('keras_loop', False, '')
flags.DEFINE_boolean('predict_on_batch', False, '')
flags.DEFINE_boolean('cache_info_dataset', False, '')
flags.DEFINE_integer('print_depth', 1, '')
flags.DEFINE_boolean('keras_functional_model', False, '')
flags.DEFINE_alias('functional_model', 'keras_functional_model')
flags.DEFINE_alias('kfm', 'keras_functional_model')
flags.DEFINE_boolean('keras_custom_progress_bar', True, '')
flags.DEFINE_alias('kcpb', 'keras_custom_progress_bar')
flags.DEFINE_bool('pb_loss_only', True, 'for tqdm progressbar only show loss realted metrics')
flags.DEFINE_bool('evshow_loss_only', True, '')
flags.DEFINE_bool('leave_epoch_progress', False, '')
flags.DEFINE_bool('leave_overall_progress', True, '')
flags.DEFINE_bool('eval_leave', True, '')
flags.DEFINE_bool('test_leave', True, '')
flags.DEFINE_bool('show_epoch_progress', True, '')
flags.DEFINE_alias('sep', 'show_epoch_progress')
flags.DEFINE_bool('show_overall_progress', True, '')
flags.DEFINE_bool('update_each_epoch', False, 'deal with None in melt.init, by default None means True when not run on notebook otherwise False')
flags.DEFINE_alias('uee', 'update_each_epoch')
flags.DEFINE_integer('keras_verbose', 1, '')
flags.DEFINE_bool('custom_loop', False, '')

flags.DEFINE_boolean('print_metrics', True, '')
flags.DEFINE_boolean('print_lr', False, '')

flags.DEFINE_boolean('test_aug', False, '')

flags.DEFINE_integer('batch_size_dim', 0, '')

flags.DEFINE_boolean('simple_parse', False, '')
flags.DEFINE_bool('write_valid', False, '')
flags.DEFINE_bool('write_valid_final', False, '')
flags.DEFINE_bool('write_valid_only', False, '')
flags.DEFINE_bool('write_valid_after_eval', False, '')
flags.DEFINE_integer('start_epoch', 0, '')
flags.DEFINE_alias('initial_epoch', 'start_epoch')
flags.DEFINE_integer('end_epoch', 0, '')
flags.DEFINE_integer('last_epoch', 1, '')

# from bert run_classifier.py
flags.DEFINE_boolean('use_tpu', False, '')
flags.DEFINE_alias('tpu', 'use_tpu')
flags.DEFINE_boolean('enable_xla', False, '')
flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
#tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# use horovod to do multiple gpu / server 
flags.DEFINE_boolean('use_horovod', False, '')
flags.DEFINE_boolean('horovod_eval', True, 'wether using multiple gpu for eval and infer, hvd.allgather not work for tf ... currently, mpi4py ok')
flags.DEFINE_boolean('horovod_shard', True, 'only consider train, valid/test always shard, if shard then each rank got 1/n files')
flags.DEFINE_boolean('horovod_fp16', False, 'wether compression use fp16 see layers/optimizer.py')
flags.DEFINE_boolean('distributed_scale', None, '') 
flags.DEFINE_boolean('shard_by_files', True, '如果不能文件均分设置False或者使用自定--use_shard=False')

# other utils

flags.DEFINE_string('lock_file', '.melt.lock', '')

flags.DEFINE_string('masked_fields', None, '')
flags.DEFINE_string('mask_mode', 'excl', '')
flags.DEFINE_integer('max_fields', None, '')

flags.DEFINE_boolean('debug', False, '')

flags.DEFINE_boolean('debug_model', False, '')
flags.DEFINE_boolean('model_summary', False, '')
flags.DEFINE_alias('summary_model', 'model_summary')

flags.DEFINE_integer('profile_interval_steps', 0, '')
flags.DEFINE_boolean('profiling', False, 'alias for enable_profiling')
flags.DEFINE_boolean('enable_profiling', False, 'for torch now')
flags.DEFINE_bool('valid_perf', False, '')

flags.DEFINE_integer('wait_gpu_span', 10, '0 means not wait')

flags.DEFINE_boolean('monitor_gradients', False, '')
flags.DEFINE_boolean('monitor_global_gradients', False, '')
flags.DEFINE_boolean("monitor_l2", True, "")

flags.DEFINE_boolean("write_summary", True, "")
flags.DEFINE_boolean("write_metric_summary", True, "")
flags.DEFINE_alias('write_metrics_summary', 'write_metric_summary')
flags.DEFINE_boolean("train_valid_summary", False, "")

flags.DEFINE_string("version", None, "")
flags.DEFINE_bool("print_version", True, "")

# ----------- distribute
flags.DEFINE_string("distribute_strategy", None, "")
flags.DEFINE_integer("ps_tasks", 0, "")

# For distributed training.
flags.DEFINE_bool('ps_strategy', False, '')
flags.DEFINE_string(
    'ps_hosts', 'localhost:8830',
    'Comma-separated list of parameter server host:port; if empty, run local')
flags.DEFINE_string(
    'worker_hosts', '', 'Comma-separated list of worker host:port')
flags.DEFINE_string(
    'job_name', '', 'The job this process will run, either "ps" or "worker"')
flags.DEFINE_integer(
    'task_index', 0, 'The task index for this process')
flags.DEFINE_integer(
    'gpu_device', 0, 'The GPU device to use.')

flags.DEFINE_boolean("tf_v2", False, "")

flags.DEFINE_boolean("allow_soft_placement", True, "")
flags.DEFINE_boolean("allow_growth", True, "")
flags.DEFINE_boolean("simple_startup", False, "")

# hack for CloudS
flags.DEFINE_boolean('sync_hdfs', True, '')
flags.DEFINE_string('hdfs_mark', 'publicData', '')
flags.DEFINE_string('local_mark', 'localData', '')

#--------------- other
flags.DEFINE_bool('is_online', False, '')
flags.DEFINE_bool('is_infer', False, 'global var to set')
flags.DEFINE_string('metric_values', '', '')
flags.DEFINE_string('metric_names', '', '')
flags.DEFINE_string('base_metric_values', '', '')
flags.DEFINE_string('base_metric_names', '', '')

# -------------- dataset specific for online training 
flags.DEFINE_string('dataset_mode', None, '')
flags.DEFINE_bool('exclude_varlen_keys', False, '')
flags.DEFINE_alias('exclude_varlen_feats', 'exclude_varlen_keys')


flags.DEFINE_bool('all_varlen_keys', False, '')
flags.DEFINE_float('dataset_rate', 1., 'only in train')

flags.DEFINE_list('dataset_keys', [], '')
flags.DEFINE_list('dataset_excl_keys', [], '')

flags.DEFINE_string('command', None, '')
flags.DEFINE_bool('async_valid', False, '')
flags.DEFINE_alias('asv', 'async_valid')
flags.DEFINE_bool('async_eval', False, 'only cpu evaluation without prediction(already done and write to csv), now only work using keras')
flags.DEFINE_bool('is_last_eval', None, '')
flags.DEFINE_alias('ase', 'async_eval')
flags.DEFINE_bool('cpu_valid', False, '')
flags.DEFINE_integer('num_valid_gpus', 1, '')

#------------------ train loop
# flags.DEFINE_bool('online_train', False, 'or use FLAGS.mode==online FLAGS.train_loop==True')
flags.DEFINE_bool('train_loop', False, '')
flags.DEFINE_alias('loop_train', 'train_loop')
flags.DEFINE_bool('mix_train', False, '')
flags.DEFINE_bool('from_loop_train', False, '')
flags.DEFINE_bool('loop_range', False, 'loop as 1,2,3...')
flags.DEFINE_bool('loop_fixed_valid', False, '')
flags.DEFINE_bool('loop_train_all', False, '')
flags.DEFINE_integer("valid_span", 1, "valid span < 0 表示另外指定一个valid_input 并且fixed valid input")
flags.DEFINE_integer("test_span", 1, "")
flags.DEFINE_integer("round", 0, "global var")
flags.DEFINE_bool('allow_round_grow', True, '')
flags.DEFINE_integer("eval_round", None, "global var")
flags.DEFINE_integer("eval_day_step", None, "global var")
flags.DEFINE_bool('eval_days', False, '')
flags.DEFINE_bool('eval_days_online', False, 'day metrics got from tfrecord as online score')
flags.DEFINE_string("loop_type", "hour", "hour or day")
flags.DEFINE_bool("loop_latest", True, "")
flags.DEFINE_string("start_hour", None, "")
flags.DEFINE_alias("start", "start_hour")
flags.DEFINE_bool('force_start', False, '')
flags.DEFINE_string("end_hour", None, "")
flags.DEFINE_alias("end", "end_hour")
flags.DEFINE_string('valid_hour', None, '')
flags.DEFINE_string('train_hour', None, '')
flags.DEFINE_boolean('sync_valid_hour', False, '')
flags.DEFINE_boolean('sync_valid_hour_final', False, '')
flags.DEFINE_integer("num_rounds", 0, "global var")
flags.DEFINE_alias('rounds', 'num_rounds')
flags.DEFINE_integer('hours', None, 'num hours(rounds) to train')
flags.DEFINE_integer('days', None, 'num days to train which means 24 * days hours(rounds)')
flags.DEFINE_integer('num_hours', None, 'num hours(rounds) to train')
flags.DEFINE_integer('num_days', None, 'num days to train which means 24 * days hours(rounds)')
flags.DEFINE_bool('dry_run', False, '')
flags.DEFINE_alias('hack_run', 'dry_run')
flags.DEFINE_alias('fake_run', 'dry_run')
flags.DEFINE_integer("valid_first_n", 1, "first n round do valid (ignore valid every n)")
# flags.DEFINE_integer("valid_after_n", 0, "valid after n round do valid (ignore valid every n)")
flags.DEFINE_integer("valid_every_n", 0, "valid every n hours(rounds), 0 means not use")
flags.DEFINE_integer("valid_every_hash_n", 0, "valid every n hours(rounds) but using hash for FLAGS.valid_hour, 0 means not use")
flags.DEFINE_integer("novalid_max_n", 0, "force to valid if not valid for nonvalid_max_n")
flags.DEFINE_bool('ev_first', False, '')
flags.DEFINE_alias('evf', 'ev_first')
flags.DEFINE_bool('ev_last', True, '')
flags.DEFINE_bool('loop_only', False, 'for eager custom loop debug only loop no model pred and loss optimize')

flags.DEFINE_bool('use_all_data', False, '')
flags.DEFINE_bool('valid_use_all_data', False, '')
flags.DEFINE_string('base_data_name', '', '')
flags.DEFINE_string('other_data_names', '', '')
flags.DEFINE_bool('use_all_type', False, '')

flags.DEFINE_bool('use_pymp', False, '')

flags.DEFINE_float('l2_', None, 'global var l2')
flags.DEFINE_integer('params_', None, 'global var params')

flags.DEFINE_integer("eval_timeout", 600, "")
flags.DEFINE_integer("del_timeout", 300, "as has watied eval_timeout so here means wating another del_timeout")
flags.DEFINE_integer("wait_gpu_timeout", 3600, "")

flags.DEFINE_integer("local_rank", 0, "")
flags.DEFINE_integer("world_size", 1, "")
flags.DEFINE_bool("distributed", False, "")
flags.DEFINE_string("distributed_command", None, "")
flags.DEFINE_bool("log_distributed", False, "")

flags.DEFINE_integer('print_precision', 4, '')

flags.DEFINE_bool('plot_graph', False, '')

flags.DEFINE_integer('num_shards', 1, '')
flags.DEFINE_string('emb_device', None, '')
flags.DEFINE_string('dataset_device', 'cpu', 'maybe set to cpu or None or empty')
flags.DEFINE_string('save_device', None, '')

flags.DEFINE_list('incl_feats', [], '')
flags.DEFINE_alias('incl', 'incl_feats')
flags.DEFINE_list('excl_feats', [], '')
flags.DEFINE_alias('excl', 'excl_feats')
flags.DEFINE_bool('re_feats', False, '')

flags.DEFINE_list('incl_keys', [], '')
flags.DEFINE_list('excl_keys', [], '')
flags.DEFINE_list('eval_keys', [], '')
flags.DEFINE_list('out_keys', [], '')
flags.DEFINE_list('str_keys', [], '')

flags.DEFINE_integer('steps_per_execution', 1, '')

flags.DEFINE_bool('write_graph', True, '')
flags.DEFINE_bool('keras_tensorboard', False, '')
flags.DEFINE_alias('ktb', 'keras_tensorboard')

flags.DEFINE_bool('dataset_ordered', False, 'here for train speed by default, not affect valid/test, for train if not shuffle then still ordered=False unless you set FLAGS.dataset_ordered=True')
flags.DEFINE_bool('use_info_dataset', False, '')

flags.DEFINE_bool('check_exists', False, '')

# for ensemble
flags.DEFINE_list('model_names', [], 'ensemble model names')
flags.DEFINE_alias('mnames', 'model_names')
flags.DEFINE_list('model_weights', [], 'ensemble weights')
flags.DEFINE_string('ensemble_activation', None, '')

flags.DEFINE_integer('num_workers', 0, 'mainly for torch dataset')

flags.DEFINE_bool('cpu_merge', False, '')

# compress
flags.DEFINE_float('initial_sparsity', 1., '')
flags.DEFINE_float('final_sparsity', 1., '')
flags.DEFINE_string('pruner_name', 'naive', '')

flags.DEFINE_bool('grad_checkpoint', False, '')

# wandb
flags.DEFINE_bool('wandb', False, '')
flags.DEFINE_string('wandb_key', None, '')
flags.DEFINE_string('wandb_project', None, '')
flags.DEFINE_string('wandb_dir', None, '')
flags.DEFINE_string('wandb_name', None, '')
flags.DEFINE_string('wandb_group', None, '')
flags.DEFINE_string('wandb_notes', None, '')
flags.DEFINE_list('wandb_tags', None, '')
flags.DEFINE_string('wandb_id', None, '')
flags.DEFINE_bool('wandb_resume', False, '')
flags.DEFINE_bool('wandb_tb', False, '开启会把tensorboard数据同步wandb,速度会变慢,图片信息也是直接写wandb更好')
flags.DEFINE_bool('wandb_magic', False, '')
flags.DEFINE_bool('wandb_quiet', False, '')
flags.DEFINE_bool('wandb_silent', None, '')
flags.DEFINE_bool('wandb_test', True, '')
flags.DEFINE_bool('wandb_image', True, '')

# kaggle
flags.DEFINE_bool('kaggle_prefix', True, '')

flags.DEFINE_bool('clear_step', False, '')

flags.DEFINE_bool('icecream', True, '')
flags.DEFINE_alias('ic', 'icecream')
flags.DEFINE_bool('icecream_context', True, '')
flags.DEFINE_alias('icc', 'icecream_context')
