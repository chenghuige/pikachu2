from tensorflow.keras.layers import Conv2D, Lambda, Dense, Multiply, Add
from tensorflow.keras import backend as K

import gezi

def elu(x, alpha=1.0):
  return K.elu(x, alpha) + 1

# ----- only a bit better.. V1
def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation=gezi.get('activation') or 'relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x

def sse_block(prevlayer, prefix):
    # Bug? Should be 1 here? But V2 below experiment change to 1 same not good result
    conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
                  activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv

def scse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x

# # # -------  V2
# def cse_block(prevlayer, prefix):
#     mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
#     lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
#     lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
#     x = Multiply()([prevlayer, lin2])
#     return x


# def sse_block(prevlayer, prefix):
#     conv = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal",
#                   activation='sigmoid', strides=(1, 1),
#                   name=prefix + "_conv")(prevlayer)
#     conv = Multiply(name=prefix + "_sse_mul")([prevlayer, conv])
#     return conv


# def scse_block(x, prefix):
#     '''
#     Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
#     https://arxiv.org/abs/1803.02579
#     '''
#     cse = cse_block(x, prefix)
#     sse = sse_block(x, prefix)
#     x = Add(name=prefix + "_scse_mul")([cse, sse])

#     return x
