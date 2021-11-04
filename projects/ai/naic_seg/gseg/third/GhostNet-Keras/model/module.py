# -- coding: utf-8 --

import math
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Concatenate, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Activation, BatchNormalization, add, Lambda 
import tensorflow.keras.backend as K
from tensorflow.keras.activations import softmax
#import tensorflow as tf

def reshapes(x):
    y = Reshape((1,1,int(x.shape[1])))(x)
    return y

def softmaxs(x):
    y = softmax(x)
    return y

def slices(x,channel):
    y = x[:,:,:,:channel] 
    return y

def multiply(x,excitation):
    scale = x * excitation
    return scale

def squeezes(x):
    y = K.squeeze(x,1)
    return y

def GhostModule(x,outchannels,ratio,convkernel,dwkernel,padding='same',strides=1,data_format='channels_last',
                use_bias=False,activation=None):
    conv_out_channel = math.ceil(outchannels*1.0/ratio)
    x = Conv2D(int(conv_out_channel),(convkernel,convkernel),strides=(strides,strides),padding=padding,data_format=data_format,
               activation=activation,use_bias=use_bias)(x)
    if(ratio==1):
        return x
    
    dw = DepthwiseConv2D(dwkernel,strides,padding=padding,depth_multiplier=ratio-1,data_format=data_format,
                         activation=activation,use_bias=use_bias)(x)
    #dw = dw[:,:,:,:int(outchannels-conv_out_channel)]
    dw = Lambda(slices,arguments={'channel':int(outchannels-conv_out_channel)})(dw)
    x = Concatenate(axis=-1)([x,dw])
    return x

def SEModule(x,outchannels,ratio):
    x1 = GlobalAveragePooling2D(data_format='channels_last')(x)
    #squeeze = Reshape((1,1,int(x1.shape[1])))(x1)
    squeeze = Lambda(reshapes)(x1)
    fc1 = Conv2D(int(outchannels/ratio),(1,1),strides=(1,1),padding='same',data_format='channels_last',
                 use_bias=False,activation=None)(squeeze)
    relu= Activation('relu')(fc1)
    fc2 = Conv2D(int(outchannels),(1,1),strides=(1,1),padding='same',data_format='channels_last',
                 use_bias=False,activation=None)(relu)
    excitation = Activation('hard_sigmoid')(fc2)
    #scale = x * excitation
    scale = Lambda(multiply,arguments={'excitation':excitation})(x)
    return scale

def GhostBottleneck(x,dwkernel,strides,exp,out,ratio,use_se):
    x1 = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',
                         activation=None,use_bias=False)(x)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Conv2D(out,(1,1),strides=(1,1),padding='same',data_format='channels_last',
               activation=None,use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    y = GhostModule(x,exp,ratio,1,3)
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    if(strides>1):
        y = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',
                         activation=None,use_bias=False)(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
    if(use_se==True):
        y = SEModule(y,exp,ratio)
    y = GhostModule(y,out,ratio,1,3)
    y = BatchNormalization(axis=-1)(y)
    y = add([x1,y])
    return y

