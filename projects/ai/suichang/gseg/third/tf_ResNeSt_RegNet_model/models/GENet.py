import os,shutil,sys
sys.path.append('../')

from copy import deepcopy
import random
import math
import numpy as np

import tensorflow as tf
# tf.enable_eager_execution()
# tf.compat.v1.enable_eager_execution()

from tensorflow.keras import optimizers,losses,activations,models,layers
from tensorflow.keras.layers import (Input,Conv1D,Conv2D,Dropout, Dense,Reshape,BatchNormalization,Activation,
                                         GlobalAveragePooling2D,GlobalAveragePooling1D, GlobalMaxPooling2D,
                                         MaxPool2D, Multiply, Add, Permute,Concatenate,Softmax,DepthwiseConv2D,
                                         ZeroPadding2D,TimeDistributed)
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.regularizers import l2

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


class Mish(Activation):
    '''
    based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    Mish Activation Function.
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    # with tf.device("CPU:0"):
    result = inputs * tf.math.tanh(tf.math.softplus(inputs))
    return result

class GENet():
    def __init__(self, verbose=False, input_shape=(224, 224, 3), active="relu", n_classes=81,
                 dropout_rate=0.2, fc_activation=None,model_name='light',using_cb=False):
        self.channel_axis = -1  # not for change
        self.verbose = verbose
        self.active = active  # default relu
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.fc_activation = fc_activation
        self.model_name = model_name

    def _make_stem(self,input_tensor,stem_width=64,deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = layers.Conv2D(stem_width,kernel_size=3,strides=2,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)

            x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            x = layers.Activation(self.active)(x)

            x = layers.Conv2D(stem_width,kernel_size=3,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)

            x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            x = layers.Activation(self.active)(x)

            x = layers.Conv2D(stem_width*2,kernel_size=3,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)

            # x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = layers.Activation(self.active)(x)
        else:
            x = layers.Conv2D(stem_width,kernel_size=7,strides=2,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)
            # x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = layers.Activation(self.active)(x)
        return x

    def _rsoftmax(self,input_tensor,filters,radix, groups):
        x = input_tensor
        batch = x.shape[0]
        if radix > 1:
            x = tf.reshape(x,[-1,groups,radix,filters//groups])
            x = tf.transpose(x,[0,2,1,3])
            x = tf.keras.activations.softmax(x,axis=1)
            x = tf.reshape(x,[-1,1,1,radix*filters])
        else:
            x = layers.Activation('sigmoid')(x)
        return x

    def _make_block_basic(self,input_tensor,filters=64,kernel_size=3, stride=1):
                        # conv_shortcut=False):
        x = input_tensor

        if stride != 1:
            shortcut = layers.Conv2D(filters, 1, padding='same',strides=stride)(x)
            shortcut = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(shortcut)
            shortcut = layers.Activation(self.active)(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters,kernel_size=3,strides=stride,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Conv2D(filters,kernel_size=3,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Add()([shortcut, x])
        return x

    def _make_block_BL(self,input_tensor,filters=64,kernel_size=3, stride=1,
                        bottleneck_ratio=0.25):
                        # conv_shortcut=False):
        x = input_tensor

        if stride != 1:
            shortcut = layers.Conv2D(filters, 1,padding='same', strides=stride)(x)
            shortcut = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(shortcut)
            shortcut = layers.Activation(self.active)(shortcut)
        else:
            shortcut = x

        current_expansion_channel = int(round(filters * bottleneck_ratio))

        x = layers.Conv2D(current_expansion_channel,kernel_size=1,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Conv2D(current_expansion_channel,kernel_size=3,strides=stride,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Conv2D(filters,kernel_size=1,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Add()([shortcut, x])
        return x

    def _make_blockX_DW(self,input_tensor,filters=64,kernel_size=3, stride=1,
                        bottleneck_ratio=3,conv_shortcut=False):
                            # groups=32,conv_shortcut=False):
        x = input_tensor
        # print('input_tensor',x.shape)

        if stride != 1 or conv_shortcut is True:
            shortcut = layers.Conv2D(filters, 1,padding='same', strides=stride)(x)
            shortcut = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(shortcut)
            shortcut = layers.Activation(self.active)(shortcut)
        else:
            shortcut = x

        current_expansion_channel = int(round(filters * bottleneck_ratio))

        x = layers.Conv2D(current_expansion_channel,kernel_size=1,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)),)(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.DepthwiseConv2D(3,strides=stride, 
                                    depth_multiplier=1,use_bias=False)(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Conv2D(filters,kernel_size=1,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)

        x = layers.Add()([shortcut, x])
        return x

    def _make_layer(self,input_tensor,filters=64,blocks=4,stride1=2,block_type='XX',bottleneck_ratio=0.25):
        x = input_tensor
        if block_type == 'XX':
            x = self._make_block_basic(x, filters,stride=stride1)
            # if blocks > 1:
            for i in range(blocks-1):
                x = self._make_block_basic(x, filters)

        elif block_type == 'BL':
            x = self._make_block_BL(x, filters,stride=stride1,bottleneck_ratio=bottleneck_ratio)
            # if blocks > 1:
            for i in range(blocks-1):
                x = self._make_block_BL(x, filters,bottleneck_ratio=bottleneck_ratio)

        elif block_type == 'DW':
            x = self._make_blockX_DW(x,filters=filters,stride=stride1,bottleneck_ratio=bottleneck_ratio,
                                        conv_shortcut=True)
            # if blocks > 1:
            for i in range(blocks-1):
                x = self._make_blockX_DW(x,filters=filters,
                                        bottleneck_ratio=bottleneck_ratio)
        else:
            raise ValueError('Unrecroginize model name {}'.format(model_name))
        return x

    def get_model_set(self,model_name):
        if model_name == 'light':
            self.channle_sets = [13,48,48,384,560,256,1920]
            self.blocks = [1,1,3,7,2,1,1]
            self.strides = [2,2,2,2,2,1,1]
        elif model_name == 'normal':
            self.channle_sets = [32,128,192,640,640,640,2560]
            self.blocks = [1,1,2,6,4,1,1]
            self.strides = [2,2,2,2,2,1,1]
        elif model_name == 'large':
            self.channle_sets = [32,128,192,640,640,640,2560]
            self.blocks = [1,1,2,6,5,4,1]
            self.strides = [2,2,2,2,2,1,1]
        else:
            raise ValueError('Unrecroginize model name {}'.format(model_name))

    def build(self):
        get_custom_objects().update({'mish': Mish(mish)})
        self.get_model_set(self.model_name)

        if self.verbose: print('model_name {}'.format(self.model_name))

        input_sig = Input(shape=self.input_shape)

        x = layers.Conv2D(self.channle_sets[0],kernel_size=3,strides=self.strides[0],
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(input_sig)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)
        if self.verbose: print('Conv 1',x.shape)

        x = self._make_layer(x,filters=self.channle_sets[1],blocks=self.blocks[1],
                                stride1=self.strides[1],block_type='XX')
        if self.verbose: print('XX 1 out',x.shape)

        x = self._make_layer(x,filters=self.channle_sets[2],blocks=self.blocks[2],
                                stride1=self.strides[2],block_type='XX')
        if self.verbose: print('XX 2 out',x.shape)

        x = self._make_layer(x,filters=self.channle_sets[3],blocks=self.blocks[3],
                                stride1=self.strides[3],block_type='BL',bottleneck_ratio=0.25)
        if self.verbose: print('BL 3 out',x.shape)

        x = self._make_layer(x,filters=self.channle_sets[4],blocks=self.blocks[4],
                                stride1=self.strides[4],block_type='DW',bottleneck_ratio=3)
        if self.verbose: print('DW 4 out',x.shape)

        x = self._make_layer(x,filters=self.channle_sets[5],blocks=self.blocks[5],
                                stride1=self.strides[5],block_type='DW',bottleneck_ratio=3)
        if self.verbose: print('DW 5 out',x.shape)

        x = layers.Conv2D(self.channle_sets[6],kernel_size=1,strides=self.strides[0],
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)
        x = layers.BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = layers.Activation(self.active)(x)
        if self.verbose: print('Conv 2',x.shape)

        fn_out = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if self.verbose: print('GlobalAveragePooling2D',fn_out.shape)

        fc_out = Dense(self.n_classes,kernel_initializer='he_normal', use_bias=False)(fn_out)

        if self.fc_activation:
            fc_out = Activation(self.fc_activation)(fc_out)

        if self.verbose: 
            print('fc_out',fc_out.shape)
        
        model = models.Model(inputs=input_sig, outputs=fc_out)

        if self.verbose: 
            print("{} builded with input {}, output{}".format(self.model_name,input_sig.shape,fc_out.shape))
            print('-------------------------------------------')
            print('')
        return model


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils.tools import load_cfg

    defaut_cfg_path = './GENet.json'
    args_config = './GENet.json'
    cfg = load_cfg(defaut_cfg_path,args_config)
    # print(cfg.__dict__)
    model = GENet(cfg,verbose=True,model_name='large').build()
    loss = [tf.keras.losses.BinaryCrossentropy(
            from_logits=False,label_smoothing=False)]
    model.compile(optimizer='adam', loss=loss)
    # print('123')
    # model = getModel.build(cfg)
    # model.summary()
