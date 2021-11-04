# -- coding: utf-8 --

import os
from .module import GhostBottleneck, reshapes ,softmaxs, squeezes
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
#from keras.activations import softmax

def GhostNet(classes=1000, input_shape=None, **kwargs):
    inputdata = Input(shape=input_shape)
    
    x = Conv2D(16,(3,3),strides=(2,2),padding='same',data_format='channels_last',activation=None,use_bias=False)(inputdata)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    # https://github.com/iamhankai/ghostnet.pytorch/issues/35
    # 我们选了 [3,5,11,16] 这些block作为FPN的输入。超参设置和你训mnasnet一样就行吧。
    # Tensor("add_2/add:0", shape=(None, 64, 64, 24), dtype=float32)
    # Tensor("add_4/add:0", shape=(None, 32, 32, 40), dtype=float32)
    # Tensor("add_10/add:0", shape=(None, 16, 16, 112), dtype=float32)
    # Tensor("add_15/add:0", shape=(None, 8, 8, 160), dtype=float32)

    x = GhostBottleneck(x,3,1,16,16,2,False)
    x = GhostBottleneck(x,3,2,48,24,2,False)
    x = GhostBottleneck(x,3,1,72,24,2,False)
    # print(x)
    # TypeError: Could not build a TypeSpec for <KerasTensor: shape=(None, 32, 32, 72) dtype=float32 (created by layer 'tf.math.multiply')> with type KerasTensor
    x = GhostBottleneck(x,5,2,72,40,2,True)
    x = GhostBottleneck(x,5,1,120,40,2,True)
    # print(x)
    x = GhostBottleneck(x,3,2,240,80,2,False)
    x = GhostBottleneck(x,3,1,200,80,2,False)
    x = GhostBottleneck(x,3,1,184,80,2,False)
    x = GhostBottleneck(x,3,1,184,80,2,False)
    x = GhostBottleneck(x,3,1,480,112,2,True)
    x = GhostBottleneck(x,3,1,672,112,2,True)
    # print(x)
    out = x

    x = GhostBottleneck(x,5,2,672,160,2,True)
    x = GhostBottleneck(x,5,1,960,160,2,False)
    x = GhostBottleneck(x,5,1,960,160,2,True)
    x = GhostBottleneck(x,5,1,960,160,2,False)
    x = GhostBottleneck(x,5,1,960,160,2,True)
    # print(x)

    x = Conv2D(960,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    #x = Reshape((1,1,int(x.shape[1])))(x)
    x = Lambda(reshapes)(x)
    x = Conv2D(1280,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Dropout(0.05)(x)
    x = Conv2D(classes,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x)
    #x = K.squeeze(x,1)
    #x = K.squeeze(x,1)
    #out = softmax(x)
    x = Lambda(squeezes)(x)
    x = Lambda(squeezes)(x)
    # out = Lambda(softmaxs)(x)
    
    
    model = Model(inputdata, out)
    #plot_model(model, to_file=os.path.join('weight', "GhostNet_model.png"), show_shapes=True)
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy']) 
    return model                 
