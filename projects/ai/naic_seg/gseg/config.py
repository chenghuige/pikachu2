#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2020-09-28 18:41:05.925247
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob

import numpy as np

import tensorflow as tf

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_list('ori_image_size', [256, 256], '')
flags.DEFINE_list('image_size', [256, 256], '280*280 for pspnet')
flags.DEFINE_list('image_size2', [256, 256], '')
flags.DEFINE_bool('dynamic_image_size', False, '')
flags.DEFINE_alias('dis', 'dynamic_image_size')
flags.DEFINE_bool('dynamic_image_scale', False, '')
flags.DEFINE_bool('dynamic_out_image_size', False, '')
flags.DEFINE_list('image_sizes', [], 'for multi scale training')
flags.DEFINE_list('image_scale_range', [], 'range of scale for image_size')

flags.DEFINE_bool('multi_scale', False, '')
flags.DEFINE_bool('multi_scale_attn', True, '')
flags.DEFINE_bool('multi_scale_attn_dynamic', False, '')
flags.DEFINE_bool('multi_scale_share', False, 'TODO')
flags.DEFINE_float('multi_scale_weight', 1., '')

# nvidia style scale attention
flags.DEFINE_integer('scale_size', 0, '')

flags.DEFINE_bool('pad_image', False, 'use pad instead of rezie when image_size not equal to ori_image_size')
flags.DEFINE_integer('show_rand_imgs', 90, '')
flags.DEFINE_integer('show_worst_imgs', 6, '')
flags.DEFINE_integer('show_best_imgs', 6, '')
flags.DEFINE_bool('show_imgs_online', False, 'True might be useful when show on notebook')
flags.DEFINE_integer('img_show_seed', 1024, '')
flags.DEFINE_bool('fast_eval', False, '')
flags.DEFINE_bool('fast_infer', False, '')
flags.DEFINE_integer('fast_infer_steps', 10, '')
flags.DEFINE_bool('interpolation', False, '')
flags.DEFINE_bool('write_image', False, 'write image to tensorboard 如果false完全不打印image 而默认True肯定至少在最后会打印')
flags.DEFINE_alias('tb_image', 'write_image')
flags.DEFINE_integer('write_image_interval', 1000000, '每隔多少个eval_step打印一次image默认等于只在最后打印')
flags.DEFINE_alias('tb_image_interval', 'write_image_interval')
flags.DEFINE_bool('write_confusion_matrix', True, '')
flags.DEFINE_bool('model_evaluate', True, 'use model.evaluate or not')
flags.DEFINE_alias('model_eval', 'model_evaluate')
flags.DEFINE_bool('custom_evaluate', False, 'use custom evaluate loop')
flags.DEFINE_alias('custom_eval', 'custom_evaluate')
flags.DEFINE_integer('custom_eval_interval', 1000000, '')
flags.DEFINE_integer('model_evaluate_count', 0, '0 means all')
flags.DEFINE_alias('mec', 'model_evaluate_count')

flags.DEFINE_bool('write_valid_results', False, 'image write to disk')
flags.DEFINE_bool('write_test_results', False, 'image write to disk')
flags.DEFINE_bool('zip_image', True, '')

flags.DEFINE_bool('write_inter_results', False, 'inter image write to disk for ensemble')

flags.DEFINE_bool('aug_train_image', True, '')
flags.DEFINE_bool('aug_pred_image', False, '')
flags.DEFINE_float('hflip_rate', 0.5, '')
flags.DEFINE_float('vflip_rate', 0.5, '0 means disable')
flags.DEFINE_float('rotate_rate', 0.5, 'rot90 prob')
flags.DEFINE_float('brightness_rate', 0., '')
flags.DEFINE_float('rgb_shift_rate', 0., '')
flags.DEFINE_float('resize_aug_rate', 0., '')
flags.DEFINE_float('color_rate', 0., '')
flags.DEFINE_float('sharpen_rate', 0., '')
flags.DEFINE_float('blur_rate', 0., '')
flags.DEFINE_float('cutmix_rate', 1., 'tested 1. current best, improve 1.5% then 0.')
flags.DEFINE_list('cutmix_range', [], '')
flags.DEFINE_float('cutmix_scale_rate', 1., '')

flags.DEFINE_string('augment_policy', None, '')
flags.DEFINE_integer('augment_level', 7, '7 now best')

flags.DEFINE_string('backbone', 'resnet50', 'resnext50 better but slow')
flags.DEFINE_string('backbone_weights', 'noisy-student', 'None or imagenet or places365 or noisy-student')
flags.DEFINE_string('backbone_dir', None, '')
flags.DEFINE_bool('backbone_trainable', True, '')

flags.DEFINE_integer('kernel_size', 3, 'final conv2d kernel size, sm 源代码写死3 这里可配置当前默认1')
flags.DEFINE_bool('preprocess', True, 'use tf.keras.applications.xx.preprocess_input')
flags.DEFINE_string('normalize_image', None, '0-1 or -1-1 注意sm.get_preprocess找不到才使用,如果找不到必须显示设置')

flags.DEFINE_bool('tta', False, '')
flags.DEFINE_list('tta_fns', ['flip_left_right', 'flip_up_down'], 'or flip_left_right,flip_up_down')
flags.DEFINE_list('tta_weights', [], 'per tta weights, notice ori image use first weight')
flags.DEFINE_bool('tta_intersect', False, '')
flags.DEFINE_bool('tta_use_original', True, '')

flags.DEFINE_bool('eval_class_per_image', True, '开始设置true在sm unet之后又做了一次conv')

flags.DEFINE_string('activation', None, '')
flags.DEFINE_bool('from_logits', True, 'from logits 效果更好')
flags.DEFINE_string('loss_fn', 'default', '')

flags.DEFINE_bool('best_now', False, '')

flags.DEFINE_bool('additional_conv', False, '')
flags.DEFINE_string('additional_conv_activation', None, '开始设置了sigmoid..')

#--- depreciated
flags.DEFINE_string('sm_conv_activation', None, '')
flags.DEFINE_integer('sm_kernel_size', 3, '')
flags.DEFINE_string('sm_decoder_block_type', 'transpose', 'or upsampling')

flags.DEFINE_float('base_lr_rate', 1., '0.01')
flags.DEFINE_float('ce_loss_rate', 0., 'if 0 means only other loss otherwise add ce_loss')

flags.DEFINE_bool('use_scse', False, '')

# post processing
flags.DEFINE_bool('post_crf', False, '')
flags.DEFINE_bool('post_remove', False, '')
flags.DEFINE_integer('min_size', 300, '')

flags.DEFINE_float('classifier_threshold', 0.1, '')
flags.DEFINE_alias('cls_thre', 'classifier_threshold')
flags.DEFINE_float('bce_loss_rate', 0., '')
flags.DEFINE_float('components_loss_rate', 0., '')

flags.DEFINE_float('multi_rate', 0., '')
flags.DEFINE_integer('multi_rate_strategy', 1, '')
flags.DEFINE_integer('mrate', 0, '')

flags.DEFINE_bool('soft_bce', False, '')

# for both unet and deeplabv3
flags.DEFINE_float('dropout', 0.3, '')
flags.DEFINE_string('inter_activation', 'swish', 'swish better then relu')

# for deeplabv3
flags.DEFINE_integer('deeplab_os', 16, '')
flags.DEFINE_float('deeplab_dropout', 0.1, '')
flags.DEFINE_string('deeplab_weights', 'cityscapes', 'pascal_voc or cityscapes')
flags.DEFINE_string('deeplab_upmethod', 'resize', '')
flags.DEFINE_list('deeplab_atrous_rates', [], '')
flags.DEFINE_bool('deeplab_sepconv', False, '')
flags.DEFINE_bool('deeplab_upsampling_last', False, 'maybe None is better but False is more convient to use not header output')
flags.DEFINE_integer('deeplab_kernel_size', 3, '')
flags.DEFINE_bool('deeplab_large_atrous', True, 'True by default, large better')
flags.DEFINE_bool('deeplab_lite', False, '')

flags.DEFINE_integer('unet_upsample_blocks', 5, '')
flags.DEFINE_bool('unet_large_filters', True, 'True by default, large better')
flags.DEFINE_list('unet_decoder_filters', [], '')
flags.DEFINE_bool('unet_use_attention', False, '')
flags.DEFINE_string('unet_skip_combiner', 'concat', '')

# for EffDet
flags.DEFINE_bool('freeze_bn', False, '')
flags.DEFINE_bool('weighted_bifpn', True, 'True效果更好 EffDet设置True和官方automl版本比差距较小 0.001 0.7843 0.7853')
flags.DEFINE_integer('head_stride', 2, '')
flags.DEFINE_integer('effdet_min_level', 3, '')
flags.DEFINE_integer('effdet_max_level', 7, '')
flags.DEFINE_integer('effdet_num_scales', 3, '')
flags.DEFINE_integer('effdet_start_level', 0, '')
flags.DEFINE_string('effdet_head_strategy', 'transpose', '')
flags.DEFINE_bool('effdet_upsampling_last', False, '')
flags.DEFINE_bool('effdet_custom_backbone', True, 'custom better metric larger model size for eff b3')
flags.DEFINE_integer('effdet_bifpn', None, '')

flags.DEFINE_integer('fpn_filters', 256, '')

flags.DEFINE_bool('adjust_preds', False, '')

flags.DEFINE_float('label_smoothing', 0., '')

flags.DEFINE_bool('official_backbone', False, 'depreciated just use offical_resnet50 official_EfficientNetB4 like this')

flags.DEFINE_string('seg_weights', None, '')

flags.DEFINE_integer('data_version', 2, '1 初赛, 2 复赛')
flags.DEFINE_string('data_src', 'naic', 'naic, baidu, ccf')

flags.DEFINE_integer('NUM_CLASSES', None, '')
flags.DEFINE_list('CLASSES', None, '')
flags.DEFINE_list('ALL_CLASSES', None, '')
flags.DEFINE_list('ROOT_CLASSES', None, '')
flags.DEFINE_bool('use_class_weights', False, '')
flags.DEFINE_list('class_weights', [], '')

flags.DEFINE_bool('classes_v2', False, 'wether use classes_v2 for data_version 1')
flags.DEFINE_bool('class_lookup', False, 'for data_version 1 which lookup to classe v2')
flags.DEFINE_bool('class_lookup_flat', False, 'lookup_flat means not lookup onehot')
flags.DEFINE_bool('class_lookup_soft', False, '')

flags.DEFINE_bool('distill', False, '')
flags.DEFINE_string('teacher', None, '')
flags.DEFINE_float('temperature', 10., '5 10?')
flags.DEFINE_float('teacher_rate', 0.5, '')
flags.DEFINE_bool('teacher_train_mode', False, '')
flags.DEFINE_integer('teacher_splits', 1, '')
flags.DEFINE_float('teacher_thre', 0., '')

flags.DEFINE_bool('onehot_dataset', False, '')
flags.DEFINE_bool('mix_dataset', False, '')
flags.DEFINE_list('dataset_weights', None, '')
flags.DEFINE_bool('dataset_loss', False, '')

flags.DEFINE_list('mlp_dims', [256, 128], '')
flags.DEFINE_bool('use_mlp', False, '')

flags.DEFINE_bool('convert', False, '')

flags.DEFINE_bool('no_labels', False, '')

flags.DEFINE_integer('swa_epochs', 10, '')
flags.DEFINE_string('swa_strategy', None, '')
flags.DEFINE_bool('swa_finetune', True, '')

flags.DEFINE_integer('weights_strategy', 0, '')

flags.DEFINE_bool('seg_lite', False, '')

import gezi
from gezi import logging
import melt as mt
from . import util

# 类别对应 data version 1
# 0 水体	1
# 1 交通运输	2
# 2 建筑	3
# 3 耕地	4
# 4 草地	5
# 5 林地	6
# 6 裸土	7
# 7 其它	8


# 复赛2级类目 data version 2
# 二级类别	对应标签的数字
# 水体	01
# 道路	02
# 建筑物	03
# 机场	04
# 火车站	05
# 光伏	06
# 停车场	07
# 操场	08
# 普通耕地	09
# 农业大棚	10
# 自然草地	11
# 绿地绿化	12
# 自然林	13
# 人工林	14
# 自然裸土	15
# 人为裸土	16
# 其它	17

# baidu  data version 3
# 0	建筑   2
# 1	耕地   3
# 2	林地   5
# 3	水体   0
# 4	道路   1
# 5	草地   4
# 6	其他   7
# 255	未标注区域  6 # will mask
# map to naic data version 1

# ccf data version 4
# 植被（标记1）、道路（标记2）、建筑（标记3）、水体（标记4）以及其他(标记0)  
# map to naic data version 1

IMAGE_SIZE = [256, 256]
# (class + 1) * 100
MATCHES = [100, 200, 300, 400, 500, 600, 700, 800]
CLASSES = ['water', 'track', 'build', 'arable', 'grass', 'forest', 'bare', 'other']
NUM_CLASSES = len(MATCHES)
ROOT_CLASSES = CLASSES
ALL_CLASSES = CLASSES

# CLASS_LOOKUP = [
#   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
# ]

## ---------------------lookup for naic data version 1 to 2
## onehot version
CLASS_LOOKUP = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

## soft version
CLASS_LOOKUP_SOFT = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #water
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #track
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #build 
  [0, 0, 0, 0, 0, 0, 0.95, 0.05, 0, 0, 0, 0, 0, 0, 0], #arable
  [0, 0, 0, 0, 0, 0, 0, 0, 0.84, 0.16, 0, 0, 0, 0, 0], #grass
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65, 0.35, 0, 0, 0], #forest
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.2, 0], #bare
  [0, 0, 0, 0, 0.711, 0.036, 0, 0, 0, 0, 0, 0, 0, 0, 0.253], #other
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 'water', 'track_road', 'build', 'track_airport', 'other_park', 'other_playground', 'arable_natural', 'arable_greenhouse',
# 'grass_natural', 'grass_greenbelt', 'forest_natural', 'forest_planted', 'bare_natural', 'bare_planted', 'other_other'
LABLE_MASK = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #water
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #track_ROAD
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #build 
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #track_airport
  [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], #ohter_park
  [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], #other_playground
  [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], #arable_natural
  [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], #arable_greenhouse
  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], #grass_natural
  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], #grass_greenbelt
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], #forest_planted
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], #bare_natural
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], #bare_planted
  [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], #other_other
]

CLASS_LOOKUP_FLAT = [0, 1, 2, 6, 8, 10, 12, 14]

# lookup for ccf data to naic data version 2

# 2020-11-03 11:14:12 2:00:06 water                |true:0.0273 pred:0.0275 acc:0.8357 recall:0.8430 iou:0.7231 fwiou:0.8867|
# 2020-11-03 11:14:12 2:00:06 track_road           |true:0.0767 pred:0.0751 acc:0.8449 recall:0.8275 iou:0.7184 fwiou:0.6760|
# 2020-11-03 11:14:12 2:00:06 build                |true:0.1212 pred:0.1256 acc:0.8876 recall:0.9194 iou:0.8235 fwiou:0.6792|
# 2020-11-03 11:14:12 2:00:06 track_airport        |true:0.0001 pred:0.0000 acc:nan recall:0.0000 iou:0.0000 fwiou:0.9991|
# 2020-11-03 11:14:12 2:00:06 other_park           |true:0.0666 pred:0.0646 acc:0.8686 recall:0.8428 iou:0.7474 fwiou:0.7478|
# 2020-11-03 11:14:12 2:00:06 other_playground     |true:0.0034 pred:0.0028 acc:0.8624 recall:0.7066 iou:0.6350 fwiou:0.9816|
# 2020-11-03 11:14:12 2:00:06 arable_natural       |true:0.2266 pred:0.2322 acc:0.8978 recall:0.9197 iou:0.8326 fwiou:0.4308|
# 2020-11-03 11:14:12 2:00:06 arable_greenhouse    |true:0.0130 pred:0.0125 acc:0.8803 recall:0.8489 iou:0.7611 fwiou:0.9535|
# 2020-11-03 11:14:12 2:00:06 grass_natural        |true:0.1275 pred:0.1234 acc:0.7187 recall:0.6956 iou:0.5468 fwiou:0.1333|
# 2020-11-03 11:14:12 2:00:06 grass_greenbelt      |true:0.0245 pred:0.0271 acc:0.7268 recall:0.8044 iou:0.6176 fwiou:0.8597|
# 2020-11-03 11:14:12 2:00:06 forest_natural       |true:0.1564 pred:0.1602 acc:0.7930 recall:0.8123 iou:0.6702 fwiou:0.2261|
# 2020-11-03 11:14:12 2:00:06 forest_planted       |true:0.0724 pred:0.0728 acc:0.7175 recall:0.7213 iou:0.5618 fwiou:0.5238|
# 2020-11-03 11:14:12 2:00:06 bare_natural         |true:0.0497 pred:0.0453 acc:0.8260 recall:0.7523 iou:0.6494 fwiou:0.7387|
# 2020-11-03 11:14:12 2:00:06 bare_planted         |true:0.0113 pred:0.0112 acc:0.7670 recall:0.7592 iou:0.6169 fwiou:0.9351|
# 2020-11-03 11:14:12 2:00:06 other_other          |true:0.0234 pred:0.0198 acc:0.6519 recall:0.5518 iou:0.4262 fwiou:0.7988|

# sm.FPN.EfficientNetB6.augl3.200epoch 配合mrate=1会OOM
# sm.Unet.eff4 配合 mrate=3会OOM

def init():
  assert FLAGS.from_logits 

  global CLASSES, NUM_CLASSES, ROOT_CLASSES, ALL_CLASSES

  if FLAGS.class_weights:
    FLAGS.class_weights = [float(x) for x in FLAGS.class_weights]

  CLASSES_V2 = ['water', 'track_road', 'build', 'track_airport', 'other_park', 'other_playground', 'arable_natural', 'arable_greenhouse',
                'grass_natural', 'grass_greenbelt', 'forest_natural', 'forest_planted', 'bare_natural', 'bare_planted', 'other_other']
  if FLAGS.data_src == 'naic':
    if FLAGS.data_version == 2:
      FLAGS.CLASSES = CLASSES_V2
      FLAGS.ALL_CLASSES = [*FLAGS.CLASSES, 'track', 'arable', 'grass', 'forest', 'bare', 'other']
      class_weights = [0.0273, 0.0767, 0.1212, 0.0001, 0.0666, 0.0034, 0.2266, 0.0130, 0.1275, 0.0245, 0.1564, 0.0724, 0.0497, 0.0113, 0.0234]
      class_weights = gezi.l1norm(class_weights)
      gezi.set('class_weights', FLAGS.class_weights or class_weights) 
      # if FLAGS.input and not 'quarter' in FLAGS.input:
      #   FLAGS.input = FLAGS.input.replace('tfrecords', 'quarter/tfrecords')
      FLAGS.class_lookup = False
      FLAGS.class_lookup_flat = False
    elif FLAGS.data_version == 1:
      if not FLAGS.classes_v2:
        FLAGS.CLASSES = CLASSES
      else:
        FLAGS.CLASSES = CLASSES_V2
        for i in range(len(CLASSES)):
          FLAGS.CLASSES[i] = CLASSES[i]
      
      if FLAGS.class_lookup or FLAGS.class_lookup_flat:
        FLAGS.CLASSES = CLASSES_V2

      if FLAGS.class_lookup_flat:
        FLAGS.vie = FLAGS.num_epochs
        print('vie set to num_epochs as model.evaluate will fail HACK')
      
      FLAGS.ALL_CLASSES = FLAGS.CLASSES
    else:
      raise ValueError(FLAGS.data_version)
  elif FLAGS.data_src == 'ccf':
    # 注意ccf baidu 都按照v1的类别映射训练 tfrecord已经做好映射 参考 prepare/
    # 植被（标记1）、道路（标记2）、建筑（标记3）、水体（标记4）以及其他(标记0)
    # FLAGS.CLASSES = ['background', 'vegetation', 'road', 'build', 'water']
    if not 'ccf_remote_dataset' in FLAGS.input:
      FLAGS.input = FLAGS.input.replace('tfrecords', 'ccf_remote_dataset/tfrecords')
    FLAGS.fold = -1 
    if FLAGS.fold is None or FLAGS.fold == -1:
      FLAGS.fold = 4
      logging.info('By default use fold 4 for ccf dataset')
  elif FLAGS.data_src == 'baidu':
    # FLAGS.CLASSES = ['build', 'arable', 'forest', 'water', 'track', 'grass', 'other', 'none']
    if not 'baidu' in FLAGS.input:
      FLAGS.input = FLAGS.input.replace('tfrecords', 'baidu/tfrecords')
  else:
    raise ValueError(FLAGS.data_src)

  if not FLAGS.CLASSES:
    FLAGS.CLASSES = CLASSES

  FLAGS.ROOT_CLASSES = FLAGS.CLASSES
  FLAGS.NUM_CLASSES = len(FLAGS.CLASSES)
    
  FLAGS.batch_parse = False
  FLAGS.static_input = True

  if FLAGS.dataset_weights:
    FLAGS.dataset_weights = list(map(float, FLAGS.dataset_weights))
  
  if FLAGS.mlp_dims:
    FLAGS.mlp_dims = list(map(float, FLAGS.mlp_dims))

  if FLAGS.deeplab_atrous_rates:
    FLAGS.deeplab_atrous_rates = list(map(int, FLAGS.deeplab_atrous_rates))

  if FLAGS.unet_decoder_filters:
    FLAGS.unet_decoder_filters = list(map(int, FLAGS.unet_decoder_filters))

  if FLAGS.cutmix_range:
    FLAGS.cutmix_range = list(map(float, FLAGS.cutmix_range))
  
  if not FLAGS.backbone_weights:
    FLAGS.backbone_weights = None

  if FLAGS.backbone_dir:
    if FLAGS.backbone_weights:
      #Automl will deal in models_factory.py
      if not FLAGS.model == 'Automl':
        bw_name = FLAGS.backbone 
        weights_path = None
        for file_  in glob.glob(f'{FLAGS.backbone_dir}/models/*'):
          name = ''.join(os.path.basename(file_).replace('-', '_').split('_'))
          if name.startswith(bw_name.split('-')[0].lower()) and FLAGS.backbone_weights.lower().split('-')[0] in name:
            weights_path = file_
            break
        FLAGS.backbone_weights = weights_path
    print('backbone weights path:', FLAGS.backbone_weights)

  if not FLAGS.additional_conv_activation:
    FLAGS.additional_conv_activation = None

  if not FLAGS.sm_conv_activation:
    FLAGS.sm_conv_activation = None

  # 当前最佳参数
  if FLAGS.best_now:
    FLAGS.kernel_size = 3
    FLAGS.optimizer = 'bert-adam'
    FLAGS.hflip_rate = 0.5
    FLAGS.vflip_rate = 0.5
    # FLAGS.rot90_rate = 0.5
    FLAGS.dropout = 0.3  
    FLAGS.label_smoothing = 0.05
    FLAGS.learning_rate = 1e-3
    FLAGS.min_learning_rate = 1e-5
    FLAGS.learning_rate_decay_power = 1.

    if FLAGS.model.startswith('bonlime'):
      FLAGS.learning_rate = 5e-4
      FLAGS.min_learning_rate = 1e-6
      FLAGS.learning_rate_decay_power = 2.

  if FLAGS.mrate:
    FLAGS.multi_rate = 0.1 #@param
    FLAGS.use_class_weights = True #@param
    FLAGS.cls_thre = 0.1 #@param
    FLAGS.multi_rate_strategy = FLAGS.mrate #@param

  if FLAGS.swa_strategy:
    # TODO 暂时写死
    GCS_ROOT = 'gs://chenghuige/data/naic2020_seg' if os.path.exists('/content') else '../input'
    # GCS_ROOT = 'gs://chenghuige/data/naic2020_seg' 
    if FLAGS.swa_strategy == 'default':
      pass
    elif FLAGS.swa_strategy == 'add_some_valid':
      FLAGS.train_input = f'{GCS_ROOT}/quarter/tfrecords/train/*/*'
      FLAGS.valid_input = f'{GCS_ROOT}/quarter/tfrecords/train/{FLAGS.fold}/2{FLAGS.fold}.*'
      FLAGS.fold = None
      FLAGS.swa_strategy = 'asv'
    elif FLAGS.swa_strategy== 'add_all_valid':
      FLAGS.train_input = f'{GCS_ROOT}/quarter/tfrecords/train/*/*'
      FLAGS.valid_input = f'{GCS_ROOT}/quarter/tfrecords/train/{FLAGS.fold}/*'
      FLAGS.allow_valid_train = True
      FLAGS.fold = None
      FLAGS.swa_strategy = 'aav'
    else:
      # other strategy teated as default
      pass

    FLAGS.swa_start = FLAGS.epochs
    FLAGS.start_epoch = FLAGS.swa_start
    # FLAGS.swa_warmup = 0.
    # FLAGS.swa_lr_ratio = 0.1
    FLAGS.ev_first = True
    FLAGS.vie = 1
    # FLAGS.swa_freq = 1
    FLAGS.epochs += FLAGS.swa_epochs
    if FLAGS.swa_finetune:
      FLAGS.clear_first = True # swa模式都是重新跑
      FLAGS.pretrain = FLAGS.mn
      FLAGS.mn += f'.swa{FLAGS.swa_epochs}_{FLAGS.swa_strategy}'

  if not FLAGS.activation:
    FLAGS.activation = None

  if FLAGS.activation:
    FLAGS.from_logits = False

  if FLAGS.backbone_weights == 'noisy-student' and not FLAGS.backbone.lower().startswith('efficient'):
    logging.info(f'Change to use backbone weights imagenet for backbone {FLAGS.backbone}')
    FLAGS.backbone_weights = 'imagenet'

  if FLAGS.backbone_weights == 'places365':
    # https://github.com/qubvel/classification_models/blob/master/classification_models/weights.py
    # for non sm usage, for sm resnet50 places365 set FLAGS.backbone_weights = 'imagenet11k-places365ch'
    backbone_weights = 'resnet50_imagenet11k-places365ch_11586_no_top.h5'
    
    if not os.path.exists(f'../input/{backbone_weights}'):
      os.system(f'wget https://github.com/qubvel/classification_models/releases/download/0.0.1/{backbone_weights} -P ../input')

    FLAGS.backbone_weights = f'../input/{backbone_weights}'

  # dataset阶段不做normalize后面Model里面preprocess函数统一处理
  assert FLAGS.preprocess

  if FLAGS.mode != 'train' and FLAGS.tta:
    FLAGS.load_graph = False

  if FLAGS.image_size and len(FLAGS.image_size) == 1:
    FLAGS.image_size.append(FLAGS.image_size[0])

  FLAGS.tta_weights = list(map(float, FLAGS.tta_weights))
  FLAGS.ori_image_size = list(map(int, FLAGS.ori_image_size))
  FLAGS.image_size = list(map(int, FLAGS.image_size))
  FLAGS.image_size2 = list(map(int, FLAGS.image_size2))
  FLAGS.image_sizes = list(map(int, FLAGS.image_sizes))
  gezi.set('image_size', FLAGS.image_size)
  FLAGS.image_scale_range = list(map(float, FLAGS.image_scale_range))

  # if FLAGS.image_sizes or FLAGS.image_scale_range:
  #   FLAGS.dynamic_image_size = True

  # deeplab这个版本效果不错 仅次于sm.Unet efficientnet 而且有较大集成diff
  # if FLAGS.model == 'bonlime.DeeplabV3Plus':
  #   FLAGS.backbone = 'xception'

  if 'xception' in FLAGS.backbone.lower():
    FLAGS.normalize_image = '-1-1'

  ## 也可以需要sm.Unet swish 在notebook手动 gezi.set('activation', 'swish')
  ## 暂时FLAGS.inter_activation 也只用在bread.DeepLabV3Plus
  # HACK now only used for sm.Unet internally
  gezi.set('activation', FLAGS.inter_activation)
  gezi.set('unet_skip_combiner', FLAGS.unet_skip_combiner)

  if FLAGS.augment_policy:
    FLAGS.aug_train_image = False

  if FLAGS.deeplab_large_atrous:
    # 默认是比较小 [6, 12, 18]
    FLAGS.deeplab_atrous_rates = [12, 24, 36]

  # PSPNet效果不好 不过测试了一下resize..
  if FLAGS.model == 'sm.PSPNet':
    if FLAGS.image_size[0] % 48 !=0:
      isize = -(-FLAGS.image_size[0] // 48)
      isize *= 48
      FLAGS.image_size = [isize, isize]

  if FLAGS.convert:
    FLAGS.save_only = True
    FLAGS.dynamic_image_size = True
    FLAGS.tta = False   # will do tta later not inside model
    FLAGS.load_weights_only = True

  if FLAGS.teacher:
    FLAGS.distill = True

  if FLAGS.distill:
    FLAGS.label_smoothing = 0.

  if FLAGS.seg_lite:
    gezi.set('seg_lite', True)

  ## 大部分可能原因都是因为image.resize是float32
  # # HACK for fp16 on tpu，目前按照是否notebook判断 实际应该判断是否tpu环境 但是tpu环境判断在mt.init 同时mt.init最后处理了fp16 需要预先设定好 TODO
  # # 有些操作如upsampling似乎不支持 fp16 另外比如FPN， 虽然fp16速度快 但是很多情况是更容易OOM 占用更多tpu mem
  # # 为了简单 和 模型存储的正确性 暂时都不使用fp16 gpu复现可以考虑打开 转换后应该没问题
  if gezi.get('tpu'):
    if not FLAGS.backbone_weights:
      FLAGS.fp16 = False #otherwise OOM batch_size 32
    if  FLAGS.sm_decoder_block_type == 'upsampling':
      FLAGS.fp16 = False  #otherwise TypeError: Value passed to parameter 'grads' has DataType bfloat16 not in list of allowed values: uint8, int8, int32, float16, float32, float64
    if 'FPN' in FLAGS.model:
      FLAGS.fp16 = False  #同上
    if FLAGS.backbone and FLAGS.backbone > 'EfficientNetB2':
      FLAGS.fp16 = False # OOM
    if FLAGS.model.startswith('efficientdet'):
      if FLAGS.image_size[0] == 512:
        FLAGS.fp16 = False  # OOM 注意即使fp16=False 大部分情况ok 跑到后半部分仍然可能OOM 原因未知
      if FLAGS.model > 'efficientdet-d2':
        FLAGS.fp16 = False # OOM

  # 注意linknet mrate=1 似乎占用tpu mem更小，特别对应eff4 size 288 如果mrate=0 OOM mrate=1可以跑batch size 32

  mt.init()

  # 放mt.init后面避免占用所有gpu
  util.get_teacher()

  logging.info('ori_image_size:', FLAGS.ori_image_size, 'image_size:', FLAGS.image_size)
  