#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   segmentation_models.py
#        \author   chenghuige  
#          \date   2020-10-04 14:42:50.442346
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import SpatialDropout2D, Conv2D
# import segmentation_models as sm
from ..third import segmentation_models as sm 
from ..config import *


# # depreciated just use Unet below
# def unet(num_classes, input_shape, backbone='resnext50', backbone_weights='imagenet',
#          backbone_trainable=True, ks=3, use_scse=False):
#   assert False, 'unet is deprecated just use Unet'
#   # https://github.com/chenghuige/kaggle_salt_bes_phalanx/blob/master/bes/models/models_zoo.py
#   kwargs = dict(
#                 decoder_block_type=FLAGS.sm_decoder_block_type, 
#                 decoder_filters=(128, 64, 32, 16, 8),
#                 decoder_use_batchnorm=True,
#                 kernel_size=FLAGS.sm_kernel_size, # 3 by fefault
#                 use_scse=use_scse,
#                 dropout=FLAGS.dropout,
#                 )

#   model = sm.Unet(
#                   input_shape=input_shape,
#                   backbone_name=backbone, 
#                   classes=num_classes, 
#                   encoder_weights=backbone_weights,
#                   encoder_freeze=not backbone_trainable, 
#                   activation=None, 
#                   **kwargs
#                  )

#   x = SpatialDropout2D(0.2)(model.output[0])

#   if FLAGS.additional_conv:
#     activation = FLAGS.additional_conv_activation
#     if ks == 1:
#       x = Conv2D(num_classes, (ks, ks), activation=activation, name="prediction")(x)
#     else:
#       x = Conv2D(num_classes, (ks, ks), padding='same', dataformat='channels_last')(x)

#   model = Model(model.input, x)

#   return model

def Unet(num_classes, input_shape, backbone='efficientnetb4', backbone_weights='imagenet',
         backbone_trainable=True, ks=3):
  # https://github.com/chenghuige/kaggle_salt_bes_phalanx/blob/master/bes/models/models_zoo.py
  kwargs = dict(
                decoder_block_type=FLAGS.sm_decoder_block_type, 
                decoder_filters=FLAGS.unet_decoder_filters or ((128, 64, 32, 16, 8) if not FLAGS.unet_large_filters else (256, 128, 64, 32, 16)),
                decoder_use_batchnorm=True,
                kernel_size=ks,
                use_scse=FLAGS.use_scse,
                use_attention=FLAGS.unet_use_attention,
                dropout=FLAGS.dropout,
                upsample_blocks=FLAGS.unet_upsample_blocks,
                )

  model = sm.Unet(
                  input_shape=input_shape,
                  backbone_name=backbone, 
                  classes=num_classes, 
                  encoder_weights=backbone_weights,
                  encoder_freeze=not backbone_trainable, 
                  activation=FLAGS.activation, 
                  **kwargs
                 )
  return model

def PSPNet(num_classes, input_shape, backbone='efficientnetb4', backbone_weights='imagenet',
         backbone_trainable=True, ks=3):
        
  model = sm.PSPNet(
                  input_shape=input_shape,
                  backbone_name=backbone, 
                  classes=num_classes, 
                  encoder_weights=backbone_weights,
                  encoder_freeze=not backbone_trainable, 
                  activation=FLAGS.activation, 
                  use_scse=FLAGS.use_scse,
                  psp_dropout=FLAGS.dropout,
                 )
  return model


def FPN(num_classes, input_shape, backbone='efficientnetb4', backbone_weights='imagenet',
         backbone_trainable=True, ks=3):
        
  model = sm.FPN(
                  input_shape=input_shape,
                  backbone_name=backbone, 
                  classes=num_classes, 
                  encoder_weights=backbone_weights,
                  encoder_freeze=not backbone_trainable, 
                  activation=FLAGS.activation, 
                  use_scse=FLAGS.use_scse,
                  pyramid_dropout=FLAGS.dropout,
                  pyramid_block_filters=FLAGS.fpn_filters,
                 )
  return model
  
def Nestnet(num_classes, input_shape, backbone='efficientnetb4', backbone_weights='imagenet',
         backbone_trainable=True, ks=3):
  # https://github.com/chenghuige/kaggle_salt_bes_phalanx/blob/master/bes/models/models_zoo.py
  kwargs = dict(
                decoder_block_type=FLAGS.sm_decoder_block_type, 
                decoder_filters=FLAGS.unet_decoder_filters or (128, 64, 32, 16, 8),
                decoder_use_batchnorm=True,
                # kernel_size=ks,
                # use_scse=use_scse,
                dropout=FLAGS.dropout,
                )

  model = sm.Nestnet(
                  input_shape=input_shape,
                  backbone_name=backbone, 
                  classes=num_classes, 
                  encoder_weights=backbone_weights,
                  encoder_freeze=not backbone_trainable, 
                  activation=FLAGS.activation, 
                  **kwargs
                 )
  return model

def Xnet(num_classes, input_shape, backbone='efficientnetb4', backbone_weights='imagenet',
         backbone_trainable=True, ks=3):
  # https://github.com/chenghuige/kaggle_salt_bes_phalanx/blob/master/bes/models/models_zoo.py
  kwargs = dict(
                decoder_block_type=FLAGS.sm_decoder_block_type, 
                decoder_filters=FLAGS.unet_decoder_filters or (128, 64, 32, 16, 8),
                decoder_use_batchnorm=True,
                # kernel_size=ks,
                # use_scse=use_scse,
                dropout=FLAGS.dropout,
                )

  model = sm.Xnet(
                  input_shape=input_shape,
                  backbone_name=backbone, 
                  classes=num_classes, 
                  encoder_weights=backbone_weights,
                  encoder_freeze=not backbone_trainable, 
                  activation=FLAGS.activation, 
                  **kwargs
                 )
  return model

def Linknet(num_classes, input_shape, backbone='efficientnetb4', backbone_weights='imagenet',
         backbone_trainable=True, ks=3):
  # https://github.com/chenghuige/kaggle_salt_bes_phalanx/blob/master/bes/models/models_zoo.py
  kwargs = dict(
                decoder_block_type=FLAGS.sm_decoder_block_type, 
                decoder_filters=FLAGS.unet_decoder_filters or ((128, 64, 32, 16, 8) if not FLAGS.unet_large_filters else (256, 128, 64, 32, 16)),
                decoder_use_batchnorm=True,
              #   kernel_size=ks,
              #   use_scse=FLAGS.use_scse,
              #   use_attention=FLAGS.unet_use_attention,
                dropout=FLAGS.dropout,
                )

  model = sm.Linknet(
                  input_shape=input_shape,
                  backbone_name=backbone, 
                  classes=num_classes, 
                  encoder_weights=backbone_weights,
                  encoder_freeze=not backbone_trainable, 
                  activation=FLAGS.activation, 
                  **kwargs
                 )
  return model
