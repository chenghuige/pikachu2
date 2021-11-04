#!/usr/bin/python3

import os
import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras import backend as K

from classification_models.tfkeras import Classifiers
import efficientnet.tfkeras as eff
import gezi
import melt as mt
from melt.image import resize

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, kernel_size2=3, rate=1, depth_activation=False, epsilon=1e-3, activation='relu'):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(activation)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)
    # bonlime version use (1,1) here seems 3,3 better for efficientnet
    x = Conv2D(filters, (kernel_size2, kernel_size2), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x

def upsampling(x, up_size, method='sampling'):
  # TODO transpose ?
  if method == 'sampling': 
    # comparing to resize sampling is faster and better  TODO scale size这里使得无法接受dynamic image input size
    # Unet,FPN 特征尺度随input image确定 最后upsampling固定的2 (segmentation_models), 比如256 输入 最后输出128 然后 再 upsampling scale 2
    scale1 = up_size[0] // x.shape[1]
    scale2 = up_size[1] // x.shape[2]
    scale_size = (scale1, scale2)
    ## 16 , 4
    # print('-----------------------scale', scale_size)  
    return tf.keras.layers.UpSampling2D(size=scale_size, interpolation='bilinear')(x)
  elif method == 'resize':
    return  Lambda(lambda x: resize(x, up_size))(x)
  else:
    raise ValueError(f'Unsupported {method}')

def AtrousSpatialPyramidPooling(input_shape, dropout=0., atrous_rates=[], activation='swish', upmethod='sampling', sepconv=False, kernel_size=3):
  if not atrous_rates:
    atrous_rates = (6, 12, 18)
  inputs = tf.keras.Input(input_shape[-3:])

  if not sepconv:
    # global pooling
    # x.shape = (batch, 1, 1, channel)
    x = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, [1,2], keepdims=True))(inputs)
    # x.shape = (batch, 1, 1, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size=(1,1), padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    # pool.shape = (batch, height, width, 256)
    pool = upsampling(x, input_shape[-3:-1], method=upmethod)
    # x.shape = (batch, height, width, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size=(1,1), dilation_rate=1, padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    dilated_1 = tf.keras.layers.Activation(activation)(x)
    # x.shape = (batch, height, width, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), dilation_rate=atrous_rates[0], padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    dilated_6 = tf.keras.layers.Activation(activation)(x)
    # x.shape = (batch, height, width, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), dilation_rate=atrous_rates[1], padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    dilated_12 = tf.keras.layers.Activation(activation)(x)
    # x.shape = (batch, height, width, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), dilation_rate=atrous_rates[2], padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    dilated_18 = tf.keras.layers.Activation(activation)(x)
    # x.shape=(batch, height, width, 256 * 5)
    x = tf.keras.layers.Concatenate(axis=-1)([pool, dilated_1, dilated_6, dilated_12, dilated_18])
    x = tf.keras.layers.Conv2D(256, kernel_size=(1,1), dilation_rate=1, padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
  else:
    # follow bonlime version
    x = inputs
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(activation)(b4)

    # upsample. have to use compat because of the option align_corners
    b4 = upsampling(b4, input_shape[-3:-1], method=upmethod)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5, kernel_size2=kernel_size)

    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5, kernel_size2=kernel_size)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5, kernel_size2=kernel_size)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)

    x = x

  return tf.keras.Model(inputs, x, name='AtrousSpatialPyramidPooling')

def _get_layernames(backbone, lite=False):
  # (16,16) (64,64)
  backbone=backbone.lower()
  if not lite:
    if backbone == 'official_resnet50':
      return ['conv4_block6_2_relu', 'conv2_block3_2_relu']
    elif backbone == 'resnet50':
      return ['stage4_unit1_relu1', 'stage2_unit1_relu1']
    elif backbone.startswith('eff'):
      return ['block6a_expand_activation', 'block3a_expand_activation']
    elif backbone == 'mobilenetv2':
      return ['block_13_expand_relu', 'block_3_expand_relu']
    else:
      raise ValueError(backbone)
  else:
    if backbone == 'resnet50':
      return ['stage3_unit1_relu1']
    elif backbone.startswith('eff'):
      return ['block4a_expand_activation']
    elif backbone == 'mobilenetv2':
      return ['block_6_expand_relu']
    else:
      raise ValueError(backbone)

def DeeplabV3Plus(input_shape, nclasses=None, weights='nosiy-student', backbone='EfficientNetB4', 
                  dropout=0., atrous_rates=[], activation='swish', upmethod='sampling', sepconv=False,
                  upsampling_last=None, kernel_size=3, lite=False):

  assert type(nclasses) is int
  
  inputs = tf.keras.Input(input_shape[-3:])

  Model = mt.image.get_classifier(backbone)
  model = Model(input_tensor=inputs, weights=weights, include_top=False)

  if gezi.get('model_summary'):
    model.summary()

  # x = model.output
  # x = Dropout(dropout)(x)
  # backbone_out = x

  ## TODO deeplab 只取两层是否太少？ 目前效果不好和取的层数 哪一层 以及dropout是否有关系？ 再对比一下pytorch版本的实现
  # a.shape=(batch, height // 4, width // 4, 256)
  layer_names = _get_layernames(backbone, lite)

  layer_name = layer_names[0]
  x = model.get_layer(layer_name).output 
  # backbone_out = x
  #  (None, 16, 16, 256)
  print('--------------1', x.shape)  
  x = AtrousSpatialPyramidPooling(x.shape[1:], dropout=dropout, atrous_rates=atrous_rates, activation=activation, sepconv=sepconv, kernel_size=kernel_size)(x)
  a = upsampling(x, [input_shape[-3] // 4, input_shape[-2] // 4], method=upmethod)

  if not lite:
    # b.shape=(batch, height // 4, width // 4, 48)
    layer_name = layer_names[1]
    x = model.get_layer(layer_name).output

    assert input_shape[-3] // 4 == x.shape[-3], f'shape1: {input_shape[-3] // 4}, shape2: {x.shape[-3]}, {input_shape}, {x.shape}'

    # tf.keras.Model(inputs, x).summary()

    # (None, 64, 64, 64)
    print('--------------2', x.shape)
    if not sepconv:
      x = tf.keras.layers.Conv2D(48, kernel_size=(1,1), padding='same', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(x)
      x = tf.keras.layers.BatchNormalization()(x)
      b = tf.keras.layers.Activation(activation)(x)
      # x.shape=(batch, height // 4, width // 4, 304)
      x = tf.keras.layers.Concatenate(axis=-1)([a, b])

      # x.shape=(batch, height // 4, width // 4, 256)
      x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation=activation, kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation(activation)(x)
      x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation=activation, kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False)(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation(activation)(x)
    else:
      x = a
      skip1 = x
      # DeepLab v.3+ decoder

      # Feature projection
      # x4 (x2) block
      
      dec_skip1 = Conv2D(48, (1, 1), padding='same',
                          use_bias=False, name='feature_projection0')(skip1)
      dec_skip1 = BatchNormalization(
          name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
      dec_skip1 = Activation(activation)(dec_skip1)
      x = Concatenate()([x, dec_skip1])

      x = SepConv_BN(x, 256, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5, kernel_size2=kernel_size)
      x = SepConv_BN(x, 256, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5, kernel_size2=kernel_size)
  
  if upsampling_last is None:
    if sepconv: # conv then upsampling
      upsampling_last = True
    else: # if not sepconv upsampling then conv
      upsampling_last = False

  if upsampling_last: # 这样conv计算量比较大 hidden_size大
    gezi.set('seg_notop', tf.keras.Model(inputs, x, name=f'DeeplabV3Plus_bread_{backbone}_notop'))
    notop_out = x
    x = tf.keras.layers.Conv2D(nclasses, kernel_size=(1,1), padding='same')(x)
    x = upsampling(x, input_shape[-3:-1], method=upmethod)
  else:
    x = upsampling(x, input_shape[-3:-1], method=upmethod)
    gezi.set('seg_notop', tf.keras.Model(inputs, x, name=f'DeeplabV3Plus_bread_{backbone}_notop'))
    notop_out = x
    x = tf.keras.layers.Conv2D(nclasses, kernel_size=(1,1), padding='same')(x)

  model = tf.keras.Model(inputs, [x, notop_out], name=f'DeeplabV3Plus_bread_{backbone}')
  # model.summary()
  return model

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  deeplabv3=DeeplabV3Plus((8,224,224,3),66)
  import numpy as np
  x=deeplabv3(tf.constant(np.random.normal(size=(8, 224, 224, 3)), dtype=tf.float32))
  deeplabv3.save('deeplabv3.h5')
  deeplabv3=tf.keras.models.load_model('deeplabv3.h5', compile=False)

