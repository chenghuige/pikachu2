#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige
#          \date   2019-07-26 23:14:46.546584
#   \Description
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi
import os
import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS
from omegaconf import OmegaConf

flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('model_mark', 8, '')
flags.DEFINE_string('data_dir', None, '')
flags.DEFINE_string('field_file_path', '../input/feat_fields.txt', '')
flags.DEFINE_string('feat_file_path', '../input/feature_index', '')
flags.DEFINE_bool('field_is_hash', True, 'field input is hash or 0-n self increase')
flags.DEFINE_string('field_lookup_container', 'array', 'if array then not using lookup hash table but using % then lookup for field index array, other wise hash')
flags.DEFINE_string('emb_activation', None, '')
flags.DEFINE_string('dense_activation', 'relu', '')
flags.DEFINE_integer('max_feat_len', 400, '')
flags.DEFINE_integer('hidden_size', 50, 'TODO should be emb dim')
flags.DEFINE_integer('other_emb_dim', None, '')

flags.DEFINE_float('doc_emb_factor', 2, '')
flags.DEFINE_float('user_emb_factor', 1, '')

flags.DEFINE_bool('use_wide_val', True, '')
flags.DEFINE_bool('use_deep_val', False, '')
flags.DEFINE_string('deep_wide_combine', 'concat', 'concat or add')
flags.DEFINE_string('pooling', 'sum', '')
flags.DEFINE_string('hpooling', None, 'history pooling')
flags.DEFINE_string('hpooling2', None, 'history pooling2')
flags.DEFINE_float('pooling_drop', 0., '')
flags.DEFINE_bool('use_history_position', False, '')

flags.DEFINE_bool('field_emb', False, 'if True better but more param')
flags.DEFINE_bool('index_addone', True, 'will not support index addone false')

flags.DEFINE_bool('rank_loss', False, '')
flags.DEFINE_bool('rank_dur_loss', False, '')

flags.DEFINE_integer('feature_dict_size', None, '')
flags.DEFINE_integer('wide_feature_dict_size', None, '')
# falgs.DEFINE_integer('doc_height', None, '')
flags.DEFINE_integer('num_feature_buckets', 3000000, '')
flags.DEFINE_integer('field_dict_size', None, '')
flags.DEFINE_integer('field_hash_size', 20000, 'make sure not hash conflict if has increase this')
flags.DEFINE_bool('need_field_lookup', False, '')
flags.DEFINE_integer('topic_dict_size', 100000, '')
flags.DEFINE_integer('keyword_dict_size', 1000000, '')
flags.DEFINE_bool("mask_use_hash", False, "offline use hash is better, but now for online pb do not support hash")

flags.DEFINE_string('mlp_dims', None, '')
flags.DEFINE_string('bot_mlp_dims', None, '')
flags.DEFINE_string('task_mlp_dims', None, '')
flags.DEFINE_float('mlp_drop', 0., '')
flags.DEFINE_bool('mlp_norm', False, '')

flags.DEFINE_bool('deep_final_act', False, 'deep do not need final act relu')

flags.DEFINE_bool('fields_attention', False, '')
flags.DEFINE_bool('onehot_fields_pooling', False, '')
flags.DEFINE_string('fields_pooling', '', '')
flags.DEFINE_string('multi_fields_pooling', '', '')
flags.DEFINE_string('fields_pooling_after_mlp', '', 'pooling after mlp then concat with mlp results')
flags.DEFINE_string('fields_pooling_after_mlp2', '', '')
flags.DEFINE_bool('use_ud_fusion', False, '')

flags.DEFINE_bool('sparse_emb', False, '')

flags.DEFINE_bool('keras_emb', True, 'for torch')
flags.DEFINE_bool('keras_linear', True, 'for torch')

flags.DEFINE_bool('use_doc_emb', False, '')
flags.DEFINE_string('doc_emb_name', 'ZDEMB', '')
flags.DEFINE_integer('doc_emb_dim', 128, '')
flags.DEFINE_bool('use_user_emb', False, '')
flags.DEFINE_bool('use_product_emb', False, '')
flags.DEFINE_bool('use_cold_emb', False, '')
flags.DEFINE_string('user_emb_name', 'ZUEMBTP', '')
flags.DEFINE_integer('user_emb_dim', 128, '')
flags.DEFINE_integer('emb_start', 2525490, '')
flags.DEFINE_float('emb_drop', 0., '')

flags.DEFINE_bool('udh_concat', False, '')

flags.DEFINE_bool('use_onehot_emb', True, '')
flags.DEFINE_bool('use_other_embs', True, '')
flags.DEFINE_bool('use_time_emb', False, '')
flags.DEFINE_bool('time_smoothing', False, '')
flags.DEFINE_integer('time_bins_per_day', None, '')
flags.DEFINE_integer('time_bins_per_hour', 6, 'by default 10 min one bin')
flags.DEFINE_bool('use_timespan_emb', False, '')
flags.DEFINE_integer('time_bin_shift_hours', 0 , '')
flags.DEFINE_bool('use_time_so', False, 'use time so or use pyfunc')

flags.DEFINE_integer('portrait_emb_dim', 366, '')
flags.DEFINE_bool('use_portrait_emb', False, '')

flags.DEFINE_bool('use_position_emb', False, '')
flags.DEFINE_bool('use_wide_position_emb', False, '')
flags.DEFINE_bool('use_deep_position_emb', False, '')
flags.DEFINE_bool('use_wd_position_emb', False, '')
flags.DEFINE_integer('num_positions', 20, 'acutally is 18 [0-17] for safe set 20 by default')
flags.DEFINE_string('position_combiner', 'concat', '')

flags.DEFINE_bool('use_history_emb', False, '')
flags.DEFINE_bool('use_kw_emb', False, '')
flags.DEFINE_bool('use_topic_emb', False, '')
flags.DEFINE_string('history_encoder', None, '')

flags.DEFINE_bool('use_title_emb', None, '')
flags.DEFINE_string('title_encoder', None, '')
flags.DEFINE_string('title_pooling', None, '')
flags.DEFINE_float('title_drop', 0., '')
flags.DEFINE_float('title_drop_rec', 0., '')
flags.DEFINE_bool('title_share_kw_emb', False, '')

flags.DEFINE_bool('use_network_emb', False, '')
flags.DEFINE_bool('use_activity_emb', False, '')
flags.DEFINE_bool('use_refresh_emb', False, '')
flags.DEFINE_bool('use_type_emb', False, '')

flags.DEFINE_bool('fields_norm', False, '')

flags.DEFINE_bool('transform', False, '')
flags.DEFINE_bool('use_label_emb', False, '')

flags.DEFINE_bool('use_weight', True, '')
flags.DEFINE_bool('duration_weight', False, '')
flags.DEFINE_bool('duration_weight_nolog', False, '')
flags.DEFINE_bool('duration_weight_obj_nolog', False, '')
flags.DEFINE_bool('click_loss_no_dur_weight', False, '')
flags.DEFINE_bool('dur_loss_no_dur_weight', False, '')
flags.DEFINE_bool('new_duration', False, '')
flags.DEFINE_float('duration_weight_power', 1., '')
flags.DEFINE_float('duration_weight_multiplier', 1., '')
flags.DEFINE_float('duration_ratio', 1., '')
flags.DEFINE_float('max_duration', 600, '')
flags.DEFINE_bool('cut_duration_by_video_time', True, '')
flags.DEFINE_integer('min_video_time', 20, '')
flags.DEFINE_float('min_finish_ratio', 0.2, '')

flags.DEFINE_bool('is_video', False, '')

flags.DEFINE_bool("interests_weight", False, "")
flags.DEFINE_string("interests_weight_type", 'drop', "ori, log, clip, drop")
flags.DEFINE_float("min_interests_weight", 0.1, "")
flags.DEFINE_integer("min_interests", 20, "20")

flags.DEFINE_bool('use_leaky', False, '')

flags.DEFINE_bool('eval_rank', True, '')
flags.DEFINE_bool('eval_group', True, '')
flags.DEFINE_bool('eval_filter_coldstart', True, '')
flags.DEFINE_string('eval_product', 'sgsapp', '')
flags.DEFINE_bool('eval_quality', True, '')
flags.DEFINE_bool('eval_cold', True, '')
flags.DEFINE_bool('eval_all', True, '')
flags.DEFINE_bool('eval_first_impression', False, '')
flags.DEFINE_bool('eval_click', True, '')
flags.DEFINE_bool('eval_dur', True, '')
flags.DEFINE_bool('eval_group_by_impression', False, '')

flags.DEFINE_string('multi_obj_type', None, 
         'None means no multi obj, simple means multi obj only from final wide and deep logits, shared bottom with 2 logits deep and 2 wide layers')
flags.DEFINE_string('multi_obj_duration_loss', 'sigmoid_cross_entropy',
                    'mean_squared_error mean_pairwise_squared_error sigmoid_cross_entropy')
flags.DEFINE_bool('use_jump_loss', False, 'if use jump loss then click_loss + dur loss after click(p(dur|click)), else click_loss + final dur loss(p(dur))')
flags.DEFINE_float('multi_obj_duration_ratio', 0.5, '')
flags.DEFINE_float('multi_obj_duration_ratio2', 0.5, '')
flags.DEFINE_bool('multi_obj_sum_loss', False, '')
flags.DEFINE_float('multi_obj_duration_infer_ratio', 0.5, '')
flags.DEFINE_string('multi_obj_merge_method', 'add', '')
flags.DEFINE_float('multi_obj_duration_infer_power', 1., '')
flags.DEFINE_float('multi_obj_jump_infer_power', 0.6, 'deprecated')
flags.DEFINE_bool('multi_obj_uncertainty_loss', False, '')
flags.DEFINE_string('click_power', '1.', 'like 1.2,1.3,1.5 for sgsapp newmse shida')
flags.DEFINE_string('dur_power', '0.6', 'like 1.2,1.3,1.5 for sgsapp newmse shida')
flags.DEFINE_string('cb_click_power', '', 'like 1.2,1.3,1.5 for sgsapp newmse shida, if not set same as click_power')
flags.DEFINE_string('cb_dur_power', '', 'like 1.2,1.3,1.5 for sgsapp newmse shida, if not set same as dur_power')
flags.DEFINE_bool('finish_ratio_as_dur', False, '')
flags.DEFINE_bool('finish_ratio_as_click', False, '')
flags.DEFINE_bool('dynamic_multi_obj_weight', False, '')
flags.DEFINE_integer('num_multi_objs', 2, 'now click and dur')
flags.DEFINE_float('multi_obj_click_max_score', None, '')
flags.DEFINE_float('multi_obj_duration_max_score', None, '')
flags.DEFINE_bool('multi_obj_auto_infer_ratio', False, '')
flags.DEFINE_bool('multi_obj_share_mlp', True, '')
flags.DEFINE_bool('compat_old_model', True, '')
flags.DEFINE_integer("num_experts", 2, "")
flags.DEFINE_string('mmoe_dims', '', '')
flags.DEFINE_bool('use_task_mlp', False, "")
flags.DEFINE_bool('logit2prob', True, '')

flags.DEFINE_bool('later_combine', False, '')

flags.DEFINE_bool('use_deep', True, '')
flags.DEFINE_integer('deep_out_dim', 1, '')
flags.DEFINE_bool('use_wide', True, '')
flags.DEFINE_bool('use_fm_first_order', False, '')

flags.DEFINE_float('duration_log_max', 8, 'default is log(600)=6.4 back to use 8')
flags.DEFINE_integer('num_duration_classes', 8, '')
flags.DEFINE_string('duration_scale', 'log', '')

flags.DEFINE_float('l2_reg', 0., '')

flags.DEFINE_bool('visual_emb', False, '')

flags.DEFINE_integer('min_click_duration', 1, '20')
flags.DEFINE_integer('min_filter_duration', 1, '5 TODO with finish ratio < 20%')

flags.DEFINE_bool('has_showtime', True, '')
flags.DEFINE_string('train_days', None, '')
flags.DEFINE_string('valid_days', None, '')

flags.DEFINE_bool('split_by_user', False, 'split train/valid by user')
flags.DEFINE_bool('split_by_time', False, '')
flags.DEFINE_bool('split_by_mix', False, '')

flags.DEFINE_bool('disable_field_emb', False, '')

flags.DEFINE_bool("finish_loss", False, "")
flags.DEFINE_float("finish_loss_ratio", 0.1, "")
flags.DEFINE_float("finish_power", 0.5, "")

flags.DEFINE_bool('compare_online', True, '')
flags.DEFINE_string('abtestids', '4,5,6', '')
flags.DEFINE_string('basetestid', None, '')

flags.DEFINE_bool('hash_encoding', False, 'input is dict based or hash based')
flags.DEFINE_string('hash_embedding_type', None, '')
flags.DEFINE_string('hash_embedding_ud_type', None, '')
flags.DEFINE_string('hash_combiner', 'sum', '')
flags.DEFINE_string('hash_combiner_ud', None, 'may use concat')
flags.DEFINE_bool('hash_append_weight', False, '')

flags.DEFINE_bool('debug_infer', False, '')

flags.DEFINE_bool('compat_old_record', False, 'compat for old tfrecords')

flags.DEFINE_bool('id_feature_only', False, '')
flags.DEFINE_bool('user_emb_only', False, '')
flags.DEFINE_bool('doc_emb_only', False, '')
flags.DEFINE_bool('deep_only', False, '')
flags.DEFINE_bool('wide_only', False, '')

flags.DEFINE_integer('concat_count', 0, '')

flags.DEFINE_bool('show_online', True, '')
flags.DEFINE_bool('eval_online', False, '')
flags.DEFINE_bool('online_result_only', False, '')
flags.DEFINE_bool('write_online_summary', False, '')
flags.DEFINE_alias('wos', 'write_online_summary')

flags.DEFINE_string('base_result_dir', None, '')

flags.DEFINE_string('cb_users', '931,984,925,926', '')
flags.DEFINE_float('cb_user_weight', 0.1, '')
flags.DEFINE_bool('change_cb_user_weight', True, 'cold start user insts will * 0.1 for neg insts only')

flags.DEFINE_bool('cold_start_infer', False, '')
flags.DEFINE_bool("valid_filter_insts", False, "we'd like to filter fo valid but it's slow so just pos filter when evaluate")

flags.DEFINE_bool('use_dense_feats', False, '')
flags.DEFINE_bool('use_other_embs_fm', False, '')

flags.DEFINE_float('min_online_auc', 0.5, '')

flags.DEFINE_bool('ignore_zero_value_feat', False, '')

flags.DEFINE_string('vars_split_strategy', 'wide_deep', 'wide_deep, emb')

flags.DEFINE_bool('filter_neg_durations', False, '')

flags.DEFINE_integer('data_version', 1, '')

flags.DEFINE_bool('split_embedding', False, 'for data_version=2, V=2 sh ..')

flags.DEFINE_bool('large_emb', False, 'if large_emb then will use multiple gpu to store')

flags.DEFINE_integer('num_heads', 2, '')
flags.DEFINE_integer('num_layers', 1, '')

flags.DEFINE_bool('use_residual', False, '')


# add by libowei
flags.DEFINE_bool('history_attention', False, '')
flags.DEFINE_bool('history_attention_v2', False, '')
flags.DEFINE_integer('history_length', 0, '')
flags.DEFINE_integer('attention_mode', 0, '')
flags.DEFINE_string('attention_mlp_dims', "20,1", '')
flags.DEFINE_string('attention_activation',"sigmoid", "")
flags.DEFINE_bool('attention_norm', True, "")

flags.DEFINE_string('history_strategy', '', '')
# -------- mkyuwen 0504
flags.DEFINE_bool('fixed_w2v_emb', False, '')  # w2v  固定emb
# flags.DEFINE_string('w2v_file_path', '', '')  # w2v  在下面
flags.DEFINE_bool('use_w2v_kw_emb', False, '')  # w2v => uniform初始化
# flags.DEFINE_bool('norm_w2v_kw_emb', False, '')  # w2v
# flags.DEFINE_bool('use_0_w2v_kw_emb', False, '')  # w2v
# flags.DEFINE_bool('use_zhengtai_w2v_kw_emb', False, '')  # w2v
# flags.DEFINE_bool('use_uniform_w2v_kw_emb', False, '')  # w2v
# mkyuwen 0612
flags.DEFINE_bool('use_split_w2v_kw_emb', False, '')  # w2v => user,doc的kw分开

flags.DEFINE_bool('use_distribution_emb', False, '')

flags.DEFINE_bool('use_merge_kw_emb', False, '')
# mkyuen 0615
flags.DEFINE_string('merge_kw_emb_pooling', 'sum', '')  # kw pooling方式
# flags.DEFINE_string('merge_kw_emb_pooling', 'avg', '')  # kw pooling方式
# flags.DEFINE_string('merge_kw_emb_pooling', 'att', '')  # kw pooling方式
# mkyuwen 0622
flags.DEFINE_bool('use_total_attn', False, '')  # attn(id,tp,kw,rec) or attn(id,tp)
# mkyuwen 0624
flags.DEFINE_bool('use_total_samekw_lbwnmktest', False, '')  # attn(kw) 与 shareKw使用相同的pooling+emb空间

flags.DEFINE_bool('use_tw_history_kw_merge_emb', False, '')
flags.DEFINE_bool('use_vd_history_kw_merge_emb', False, '')
flags.DEFINE_bool('use_rel_vd_history_kw_merge_emb', False, '')
flags.DEFINE_bool('use_doc_kw_merge_emb', False, '')
flags.DEFINE_bool('use_doc_kw_secondary_merge_emb', False, '')
flags.DEFINE_bool('use_tw_long_term_kw_merge_emb', False, '') 
flags.DEFINE_bool('use_vd_long_term_kw_merge_emb', False, '')
flags.DEFINE_bool('use_new_search_kw_merge_emb', False, '')
flags.DEFINE_bool('use_long_search_kw_merge_emb', False, '')
flags.DEFINE_bool('use_user_kw_merge_emb', False, '')

flags.DEFINE_bool('use_kw_merge_score', False, '')
flags.DEFINE_bool('use_kw_secondary_merge_score', False, '')
flags.DEFINE_bool('use_new_search_kw_merge_score', False, '')
flags.DEFINE_bool('use_user_kw_merge_score', False, '')

import melt
import gezi
logging = gezi.logging

# NOTICE called before melt.init .. so with out any melt.inited.. HACK might need also init from env here (duplicate TODO)
def init():     
  FLAGS.recount_tfrecords = True

  FLAGS.emb_device = FLAGS.emb_device or FLAGS.hack_device

  root = os.path.dirname(os.path.abspath(__file__))
  
  if not FLAGS.use_onehot_emb:
    FLAGS.deep_only = True

  if FLAGS.data_version != 1:
    assert gezi.get_env('V') == str(FLAGS.data_version), f'V={FLAGS.data_version} sh ..'

  if 'ABTESTIDS' in os.environ:
    FLAGS.abtestids = os.environ['ABTESTIDS']
    
  if 'DIR' in os.environ:
    FLAGS.data_dir = os.environ['DIR']

  if 'NO_MODEL_SUFFIX' in os.environ and os.environ['NO_MODEL_SUFFIX'] == '1':
    FLAGS.disable_model_suffix = True
    
  if 'ONLINE' in os.environ and os.environ["ONLINE"] == "1": 
    FLAGS.min_click_duration = 20
    FLAGS.min_interests = 30
    FLAGS.model_dir += '.online'

  if 'INTERESTS_WEIGHT' in os.environ:
    FLAGS.interests_weight = bool(int(os.environ['INTERESTS_WEIGHT']))
    
  if 'FIELD_EMB' in os.environ:
    FLAGS.field_emb = bool(int(os.environ['FIELD_EMB']))
  
  if FLAGS.data_dir:
    FLAGS.field_file_path = os.path.join(FLAGS.data_dir, 'fields.txt')
  
    if FLAGS.train_input and '../input' in FLAGS.train_input:
      train_input = FLAGS.train_input.replace('../input', FLAGS.data_dir)
      if gezi.list_files(train_input):
        FLAGS.train_input = train_input
    if FLAGS.valid_input and '../input' in FLAGS.valid_input:
      valid_input = FLAGS.valid_input.replace('../input', FLAGS.data_dir)
      if gezi.list_files(valid_input):
        FLAGS.valid_input = valid_input
    if FLAGS.model_dir and '../input' in FLAGS.model_dir:
      FLAGS.model_dir = FLAGS.model_dir.replace('../input', FLAGS.data_dir)
    if not FLAGS.train_input:
      FLAGS.train_input = os.path.join(FLAGS.data_dir, 'tfrecord/train')
    if not FLAGS.valid_input:
      FLAGS.valid_input = os.path.join(FLAGS.data_dir, 'tfrecord/valid') 
    if not FLAGS.model_dir:
      FLAGS.model_dir = os.path.join(FLAGS.data_dir, 'model/tmp') 
  else:
    FLAGS.data_dir = f'{root}/../input'

  if 'video' in FLAGS.model_dir or 'video' in FLAGS.data_dir: 
    FLAGS.is_video = True

  mark = 'tuwen' if not FLAGS.is_video else 'video'
  FLAGS.field_file_path = f'{root}/conf/{mark}/fields.txt'
  if FLAGS.use_all_type:
    FLAGS.field_file_path = FLAGS.field_file_path.replace('tuwen', 'video')
    FLAGS.use_type_emb = True
        
  if gezi.has_env('TRAIN_INPUT'):
    FLAGS.train_input = os.environ['TRAIN_INPUT']
  if gezi.has_env('VALID_INPUT'):
    FLAGS.valid_input = os.environ['VALID_INPUT']
     
  if FLAGS.use_wide_position_emb or FLAGS.use_deep_position_emb or FLAGS.use_wd_position_emb:
    FLAGS.use_position_emb = True
    
  if FLAGS.id_feature_only:
    FLAGS.use_user_emb = True 
    FLAGS.use_doc_emb = True
    FLAGS.deep_only = True
    
  if FLAGS.user_emb_only:
    FLAGS.use_doc_emb = False 
    
  if FLAGS.doc_emb_only:
    FLAGS.use_user_emb =False

  if (FLAGS.deep_only or FLAGS.wide_only) and FLAGS.vars_split_strategy == 'wide_deep':
    if FLAGS.num_optimizers is None:
      FLAGS.num_optimizers = 1

  if FLAGS.use_history_emb:
    FLAGS.use_doc_emb = True
  
  if FLAGS.valid_filter_insts:
    FLAGS.batch_parse = False

  if FLAGS.base_result_dir and FLAGS.basetestid:
    # xx/15 -> xx/16
    FLAGS.base_result_dir = os.path.join(os.path.dirname(FLAGS.base_result_dir), FLAGS.basetestid) 
    
  if FLAGS.fields_pooling or FLAGS.field_emb:
    FLAGS.need_field_lookup = True
  
  if ',' not in FLAGS.click_power:
    FLAGS.click_power = ','.join([FLAGS.click_power] * 5)
    FLAGS.dur_power = ','.join([FLAGS.dur_power] * 5)
    
  FLAGS.cb_click_power = FLAGS.cb_click_power or FLAGS.click_power
  FLAGS.cb_dur_power = FLAGS.cb_dur_power or FLAGS.dur_power

  FLAGS.hash_combiner_ud = FLAGS.hash_combiner_ud or FLAGS.hash_combiner
  FLAGS.hash_embedding_ud_type = FLAGS.hash_embedding_ud_type or FLAGS.hash_embedding_type
  FLAGS.wide_feature_dict_size = FLAGS.wide_feature_dict_size or FLAGS.feature_dict_size
  # FLAGS.doc_height = FLAGS.doc_height or FLAGS.feature_dict_size
  FLAGS.task_mlp_dims = FLAGS.task_mlp_dims or FLAGS.mlp_dims
  FLAGS.hpooling = FLAGS.hpooling or FLAGS.pooling
  FLAGS.other_emb_dim = FLAGS.other_emb_dim or FLAGS.hidden_size

  # add 2 for padding zero and unknown 1
  if not FLAGS.masked_fields:
    if os.path.exists(FLAGS.field_file_path):
      FLAGS.field_dict_size = len(set([line.strip().split()[-1] for line in open(FLAGS.field_file_path) if line.strip()])) + 1

  embedding_dims_file = f'{root}/conf/{mark}/embedding_dims.txt' 
  if os.path.exists(embedding_dims_file):
    input_dims = {}
    output_dims = {}
    total_count = 0
    total_input_dim = 0
    for line in open(embedding_dims_file):
      l = line.strip().split()
      if len(l) < 2:
        continue
      total_count += int(l[1])
    for line in open(embedding_dims_file):
      l = line.strip().split()
      if len(l) < 2:
        continue
      field = l[0]
      count = int(l[1])
      if count > 200:
        input_dim = int(FLAGS.num_feature_buckets * (count / total_count)) + 1
        if input_dim < 10000:
          input_dim = 20001
      else:
        input_dim = count * 100
      input_dims[field] = input_dim
      total_input_dim += input_dim
      if len(l) > 2:
        output_dim = l[2]
        output_dims[field] = output_dim
      logging.debug(field, count, count / total_count, input_dim, total_input_dim)
    gezi.set_global('embedding_input_dims', input_dims)
    gezi.set_global('embedding_output_dims', output_dims)

  embedding_infos_file = f'{root}/conf/{mark}/embedding_infos.yaml'
  if os.path.exists(embedding_infos_file):
    embedding_infos = OmegaConf.load(embedding_infos_file)
    gezi.set_global('embedding_infos', embedding_infos)

  if FLAGS.padded_tfrecord:
    varlens = {}
    mark = 'video' if FLAGS.is_video else 'tuwen'
    for line in open(f'./conf/{mark}/varlen.txt'):
      l = line.strip().split()
      key, val = l[0], l[1]
      val = int(l[1])
      varlens[key] = val
    gezi.set('varlens', varlens)
  
  melt.init()
