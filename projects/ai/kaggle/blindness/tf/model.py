# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model class for Cifar10 Dataset."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_base

import melt


class BaseModel(model_base.ResNet):

  def __init__(self,
               num_layers,
               training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_first'):
    super(BaseModel, self).__init__(
        training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 10 + 1
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    
  def init_predict(self, input_data_format='channels_last'):
    #self.image_feed = tf.placeholder_with_default(tf.constant([test_image]), [None, ], name='image_feature')
    self.image_feed =  tf.compat.v1.placeholder(tf.string, [None,], name='image')
    tf.compat.v1.add_to_collection('feed', self.image_feed)
    image = tf.map_fn(lambda img: melt.image.decode_image(img, image_format='png', dtype=tf.float32),
                      self.image_feed, dtype=tf.float32)
    self.predict(image)
    tf.compat.v1.add_to_collection('classes', self.pred['classes'])
    tf.compat.v1.add_to_collection('probabilities', self.pred['probabilities'])
    tf.compat.v1.add_to_collection('logits', self.logits)
    tf.compat.v1.add_to_collection('pre_logits', self.pre_logits)

  def forward_pass(self, x, input_data_format='channels_last'):
    # TODO.. without this forward var scope inference_fn will cause problem for self._conv as try to add conv 43.. FIMXE
    with tf.compat.v1.variable_scope('forward', reuse=tf.compat.v1.AUTO_REUSE):
      """Build the core model within the graph."""
      if self._data_format != input_data_format:
        if input_data_format == 'channels_last':
          # Computation requires channels_first.
          x = tf.transpose(a=x, perm=[0, 3, 1, 2])
        else:
          # Computation requires channels_last.
          x = tf.transpose(a=x, perm=[0, 2, 3, 1])

      # Image standardization.
      x = x / 128 - 1

      x = self._conv(x, 3, 16, 1)
      x = self._batch_norm(x)
      x = self._relu(x)

      # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
      res_func = self._residual_v1

      # 3 stages of block stacking.
      for i in range(3):
        with tf.compat.v1.name_scope('stage'):
          for j in range(self.n):
            if j == 0:
              # First block in a stage, filters and strides may change.
              x = res_func(x, 3, self.filters[i], self.filters[i + 1],
                          self.strides[i])
            else:
              # Following blocks in a stage, constant filters and unit stride.
              x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)

      x = self._global_avg_pool(x)
      self.pre_logits = x 
      
      x = self._fully_connected(x, self.num_classes)

      self.logits = x

      return x

  def predict(self, x=None, input_data_format='channels_last'):
    if x is not None:
      self.forward_pass(x, input_data_format)
    
    logits = self.logits
    pred = {
      'classes': tf.cast(tf.argmax(input=logits, axis=1), dtype=tf.int32),
      'probabilities': tf.nn.softmax(logits)
    }

    self.pred = pred

    return pred

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
     
    with tf.compat.v1.variable_scope('resnet', reuse=tf.compat.v1.AUTO_REUSE) as scope:
      self.models = [None] * 2
      data_format = 'channels_first'
      num_layers = 44
      batch_norm_decay = 0.997
      batch_norm_epsilon = 1e-05
      data_dir = './mount/data/cifar10/' 
      for i in range(2):
        self.models[i] = BaseModel(
                              num_layers,
                              batch_norm_decay=batch_norm_decay,
                              batch_norm_epsilon=batch_norm_epsilon,
                              training=(i == 1),
                              data_format=data_format)         
  
  def call(self, x, input_data_format='channels_last', training=False):
    x = x['image']
    model = self.models[int(training)]
    return model.forward_pass(x, input_data_format)

