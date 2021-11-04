# -*- coding: utf-8 -*-
"""
Implementation of Fast-SCNN using Tensorflow 2.0 and TF.Keras

Official paper link: https://arxiv.org/abs/1902.04502
Author: kshitizrimal
GitHub: https://github.com/kshitizrimal/Fast-SCNN

Slightly modified for use to enable the calling of the model
from the python file. Mostly moved things around so they sit inside of 
a function.

Also noticed that no weight decay was used? As per the original article
its a weight decay (or l2 regularizer) of 0.00004 for standard convolutions
with none applied to depthwise convolutions.
"""


import tensorflow as tf
from tensorflow import keras
import gezi

#### Custom function for conv2d: conv_block
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True, reg=keras.regularizers.l2(0.00004), activation='relu'):
  
  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides, kernel_regularizer=reg, bias_regularizer=reg)(inputs)  
  
  x = tf.keras.layers.BatchNormalization()(x)
  
  if (relu):
    x = tf.keras.layers.Activation(activation)(x)
  
  return x

#### residual custom method
def _res_bottleneck(inputs, filters, kernel, t, s, r=False, activation='relu'):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n, activation='relu'):
  x = _res_bottleneck(inputs, filters, kernel, t, strides, activation=activation)
  
  for i in range(1, n):
    x = _res_bottleneck(x, filters, kernel, t, 1, True, activation=activation)

  return x    

def pyramid_pooling_block(input_tensor, bin_sizes, input_size):
  concat_list = [input_tensor]
  w = input_size[0] // 32
  h = input_size[1] // 32

  for bin_size in bin_sizes:
    x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same',kernel_regularizer=keras.regularizers.l2(0.00004), bias_regularizer=keras.regularizers.l2(0.00004))(x)
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

    concat_list.append(x)

  return tf.keras.layers.concatenate(concat_list)

def model(input_size, num_classes, activation='relu'):

  # Input Layer
  input_layer = tf.keras.layers.Input(shape=input_size, name = 'input_layer')

  ## Step 1: Learning to DownSample
  lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
  lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
  lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))

  ## Step 2: Global Feature Extractor
  gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3, activation=activation)
  gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3, activation=activation)
  gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3, activation=activation)
  gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8], input_size)

  """## Step 3: Feature Fusion"""
  ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False, activation=activation)

  ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
  ff_layer2 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
  ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
  ff_layer2 = tf.keras.layers.Activation(activation)(ff_layer2)
  ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None,
                                     kernel_regularizer=keras.regularizers.l2(0.00004), 
                                     bias_regularizer=keras.regularizers.l2(0.00004))(ff_layer2)

  ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
  ff_final = tf.keras.layers.BatchNormalization()(ff_final)
  ff_final = tf.keras.layers.Activation(activation)(ff_final)

  """## Step 4: Classifier"""

  classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  classifier = tf.keras.layers.Activation(activation)(classifier)

  classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  classifier = tf.keras.layers.Activation(activation)(classifier)

  if not gezi.get('seg_notop'):
    gezi.set('seg_notop', tf.keras.Model(input_layer, classifier, name='Fast_SCNN_notop'))
    seg_notop_out = classifier

  classifier = conv_block(classifier, 'conv', num_classes, (1, 1), strides=(1, 1), padding='same', relu=False)

  classifier = tf.keras.layers.Dropout(0.3)(classifier)

  classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
  #Since its likely that mixed precision training is used, make sure softmax is float32
  classifier = tf.dtypes.cast(classifier, tf.float32)
  # classifier = tf.keras.activations.softmax(classifier)

  fast_scnn = tf.keras.Model(inputs=input_layer, outputs=classifier, name='Fast_SCNN')

  return fast_scnn
