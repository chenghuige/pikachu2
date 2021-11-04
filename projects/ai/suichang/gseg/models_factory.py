#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   models_factory.py
#        \author   chenghuige  
#          \date   2020-11-04 15:09:02.618618
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from gezi import logging
import melt as mt
from .config import  *
from .util import *

def get_model(model_name, input_shape=None):
  NUM_CLASSES = FLAGS.NUM_CLASSES
  n_ch = 3 if not FLAGS.use_nir else 4
  input_shape = input_shape or (*FLAGS.image_size, n_ch) if not FLAGS.dynamic_image_size else (None, None, n_ch)
  if is_classifier():
    Model = mt.image.get_classifier(FLAGS.backbone, FLAGS.official_backbone)
    model = Model(input_shape=input_shape, include_top=False)

  elif model_name.startswith('base'):  # get best 69.7
    from .models.baseline import unet
    model = unet(NUM_CLASSES, input_shape, ks=FLAGS.kernel_size, activation=FLAGS.activation)
  
  elif model_name == 'unet':
    from .models import unet
    Model = unet.get_model(FLAGS.backbone)
    model = Model(NUM_CLASSES, input_shape, dropout=FLAGS.dropout, weights=FLAGS.backbone_weights)
  
  elif model_name.startswith('sm.'):  # unet tested good results efficientnetb4 best 79.57, other models have not tested yet
    from .models import seg_models as models
    mname = model_name.split('.')[-1]
    Model = getattr(models, mname)
    kwargs = {
      'backbone': FLAGS.backbone.lower(),
      'backbone_weights': FLAGS.backbone_weights,
      'backbone_trainable': FLAGS.backbone_trainable,
      'ks': FLAGS.kernel_size,
    }
   
    model = Model(NUM_CLASSES, input_shape, **kwargs)
  
  elif model_name == 'uefficientnet':
    pass
  
  elif model_name.startswith('tachi'):  # bad result deeplabv3+ not as good as bread version and also slow
    from .third.TachibanaYoshino import models
    mname = model_name.split('.')[-1]
    Model = getattr(models, mname)
    if mname == 'unet':
      input = Input(shape=input_shape)
      pred = Model(NUM_CLASSES, input)
      model = keras.Model(input, pred)
    else:
      # default DeepLabV3Plus & ResNet152
      model = Model(NUM_CLASSES, base_model=FLAGS.backbone)(input_size=tuple(FLAGS.image_size))
  
  elif model_name.startswith('bread'):  # deeplabv3+ good result, now only resnet50, tested with official resnet and sm resnet with places365 pretrain
    from .third.breadbread1984 import Model as models
    # bread.DeeplabV3Plus
    try:
      mname = model_name.split('.')[-1]
      Model = getattr(models, mname) 
    except Exception:
      Model = models.DeeplabV3Plus
    model = Model(input_shape, NUM_CLASSES, weights=FLAGS.backbone_weights, 
                  backbone=FLAGS.backbone, dropout=FLAGS.deeplab_dropout, 
                  activation=FLAGS.inter_activation, atrous_rates=FLAGS.deeplab_atrous_rates, 
                  upmethod=FLAGS.deeplab_upmethod, sepconv=FLAGS.deeplab_sepconv,
                  upsampling_last=FLAGS.deeplab_upsampling_last, 
                  kernel_size=FLAGS.deeplab_kernel_size,
                  lite=FLAGS.deeplab_lite)
  
  elif model_name.startswith('bonlime'):
    from .third.bonlime.model import Deeplabv3
    weights = FLAGS.deeplab_weights if FLAGS.backbone in ['xception', 'mobilenetv2'] else FLAGS.backbone_weights
    model = Deeplabv3(input_shape=input_shape, classes=NUM_CLASSES, 
                      backbone=FLAGS.backbone, weights=weights,
                      OS=FLAGS.deeplab_os, dropout=FLAGS.dropout, 
                      lite=FLAGS.deeplab_lite)
  
  elif model_name == 'official': # mobilenet + unet not good
    from .third.official.model import unet_model
    model = unet_model(NUM_CLASSES)

  elif model_name.lower() == 'fast_scnn':
    from .third.SkyWa7ch3r.fast_scnn import model as FastSCNN
    FLAGS.backbone = None
    FLAGS.normalize_image = FLAGS.normalize_image or '0-1'
    model = FastSCNN(input_shape, NUM_CLASSES, activation=FLAGS.inter_activation)

  elif model_name.lower() == 'hrnet':
    from .third.nicongchong.hrnet.model.seg_hrnet import seg_hrnet
    model = seg_hrnet(input_shape, NUM_CLASSES)

  elif model_name.lower() == 'hrnetv2':
    from .third.yinguobing.network import hrnet_v2
    model = hrnet_v2(input_shape, NUM_CLASSES)

  elif model_name.lower() == 'refinenet':
    from .third.Attila94.model.refinenet import build_refinenet
    model = build_refinenet(input_shape, NUM_CLASSES, backbone=FLAGS.backbone, weights=FLAGS.backbone_weights)

  # 官方版本 EfficientDet (automl)
  elif model_name.lower()  == 'automl':
    ## 注意当前只能save checkpoint, save h5会报错 有的layer no name FIXME
    ## seems effdet has no preprocess inside, as show in https://github.com/google/automl/blob/master/efficientdet/keras/segmentation.py you need to prprocess before feeding
    sys.path.append(os.path.join(os.path.dirname(__file__), 'third/automl/efficientdet'))
    import hparams_config
    # from keras import efficientdet_keras
    from .third.automl.efficientdet.keras import efficientdet_keras
    
    ## efficientdet-lite0 efficientdet-d0 - efficientdet-d7, input backbone can be EfficientNetB0, EffficientNetlite0
    def _convert(backbone_name):
      model_name = backbone_name[:len('EfficientNet')].replace('EfficientNet', 'efficientdet') + '-' + backbone_name[len('EfficientNet'):].replace('B', 'd')
      return model_name
    
    model_name = _convert(FLAGS.backbone)
    if 'lite' in model_name:
      FLAGS.backbone_weights = None

    config = hparams_config.get_efficientdet_config(model_name)
    config.heads = ['segmentation']
    # config.heads = ['object_detection']
    config.seg_num_classes = NUM_CLASSES
    config.min_level = FLAGS.effdet_min_level # effdet 默认3 这时候和 Unet相比其实差两个level Unet想当 min_level=1, minlevel 越低 fpn feats 越增加后面的比如 2 增加 64  1 增加 64 128
    config.max_level = FLAGS.effdet_max_level # 这个不起作用
    config.num_scales = FLAGS.effdet_num_scales
    gezi.set('head_stride', FLAGS.head_stride)
    gezi.set('effdet_start_level', FLAGS.effdet_start_level)
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.build((1, *FLAGS.image_size, 3))
    # print('-----------', model.layers[-2])

    # TODO load 不成功。。。 
    # ValueError: Layer #3 (named "fpn_cells"), weight <tf.Variable 'fpn_cells/cell_0/fnode0/WSM:0' shape=() dtype=float32, numpy=1.0> has shape (), but the saved weight has shape (3, 3, 112, 1).
    if FLAGS.backbone_weights and os.path.exists(FLAGS.backbone_weights):
      model.load_weights(FLAGS.backbone_weights, by_name=True, skip_mismatch=True)
    elif FLAGS.backbone_weights:
      if not FLAGS.backbone_dir:
        # https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.h5  0-7,7x
        # https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d2-640.h5  2-6
        if FLAGS.backbone_weights == '640':
          NS_WEIGHTS_PATH = 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/'
          file_name = f'{model_name}-640.h5'
        else:
          NS_WEIGHTS_PATH = 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/'
          file_name = f'{model_name}.h5'
        weights_path = tf.keras.utils.get_file(
                file_name,
                NS_WEIGHTS_PATH + file_name,
            )
      else:
        if FLAGS.backbone_weights == '640':
          file_name = f'{model_name}-640.h5'
        else:
          file_name = f'{model_name}.h5'
        weights_path = f'{FLAGS.backbone_dir}/datasets/{file_name}'

      ## 这个不行。。。 model.h5 不可以 如果不是by_name layer少一层 如果是by_name 
      ## ValueError: Layer #3 (named "fpn_cells"), weight <AutoCastVariable 'fpn_cells/cell_0/fnode0/WSM:0' shape=() dtype=float32 true_dtype=float32, numpy=1.0> has shape (), but the saved weight has shape (3, 3, 64, 1).
      ## HACK 暂时都改了名字 
      mt.l2_info(model, f'before loading {model_name} backbone:')
      # model.load_weights(weights_path, by_name=FLAGS.load_by_name)
      model.load_weights(weights_path, by_name=True, skip_mismatch=True)
      mt.l2_info(model, f'after loading {model_name} backbone:')
      # model.load_weights(tf.train.latest_checkpoint('/home/featurize/data/efficientdet-d0'))
  
  # 非官方keras版本EfficientDet
  elif model_name == 'EfficientDet':
    phi = int(FLAGS.backbone[-1])
    from .third.EfficientDet.model import efficientdet
    model = efficientdet(phi, input_shape, NUM_CLASSES, freeze_bn=FLAGS.freeze_bn, 
                         weighted_bifpn=FLAGS.weighted_bifpn, 
                         weights=FLAGS.backbone_weights, 
                         head_strategy=FLAGS.effdet_head_strategy, 
                         upsampling_last=FLAGS.effdet_upsampling_last,
                         start_level=FLAGS.effdet_start_level, 
                         bifpn=FLAGS.effdet_bifpn,
                         custom_backbone=FLAGS.effdet_custom_backbone)
#     if FLAGS.scale_size:
#       # TODO 似乎比较难弄 如何共享backbone ？ 函数接口似乎不是很方便？
#       model2 = efficientdet(phi, input_shape, NUM_CLASSES, freeze_bn=FLAGS.freeze_bn, 
#                           weighted_bifpn=FLAGS.weighted_bifpn, 
#                           weights=FLAGS.backbone_weights, 
#                           head_strategy=FLAGS.effdet_head_strategy, 
#                           upsampling_last=FLAGS.effdet_upsampling_last)
#       gezi.set('model2', model2)

  elif model_name == 'mnasnetfc':
    from .third.mnasnet_fc.nets.MnasnetEager import MnasnetFC
    model = MnasnetFC(NUM_CLASSES, input_shape=input_shape, name='mnasnetfc')

  else:
    raise ValueError(f'No {model_name}')
  
  logging.info('model:', model, 'backbone:', FLAGS.backbone)
  return model