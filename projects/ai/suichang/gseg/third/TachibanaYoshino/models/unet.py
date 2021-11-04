import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Convolution2DTranspose, Concatenate
from tensorflow.keras import backend as K


def conv_block(input_tensor, filters, kernel_size, name, strides, padding='same', dila=1):
    x = Conv2D(filters, kernel_size, strides=strides, name=name, padding=padding,
               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5), dilation_rate=dila)(input_tensor)
    x = BatchNormalization(name='bn_' + name)(x)
    x = Activation('relu')(x)
    return x

# -----


def Net(n_classes, input):

    # ---------left branch -----
    x = conv_block(input, 32, (3, 3), strides=1, name='L_conv1-1')
    L1 = conv_block(x, 32, (3, 3), strides=1, name='L_conv1-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L1)
    #   256 -> 128

    x = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-1')
    L2 = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L2)
    #   128 -> 64

    x = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-1')
    L3 = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L3)
    #   64 -> 32

    x = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-1')
    L4 = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L4)
    #   32 -> 16

    x = conv_block(x, 512, (3, 3), strides=1, name='bottom-1')
    x = conv_block(x, 512, (3, 3), strides=1, dila=2, name='bottom-2')
    L5 = conv_block(x, 512, (3, 3), strides=1, name='bottom-3')
    #    16

    # ---------Right branch -----

    #   16 -> 32
    x = Convolution2DTranspose(
        256, kernel_size=2, strides=2, padding='same', name='R_conv1-1')(L5)
    x = BatchNormalization(name='R_conv1-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L4]), 256, (3, 3), strides=1, name='R_conv1-2')
    x = conv_block(x, 256, (3, 3), strides=1, name='R_conv1-3')

    #   32 -> 64
    x = Convolution2DTranspose(
        128, kernel_size=2, strides=2, padding='same', name='R_conv2-1')(x)
    x = BatchNormalization(name='R_conv2-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L3]), 128, (3, 3), strides=1, name='R_conv2-2')
    x = conv_block(x, 128, (3, 3), strides=1, name='R_conv2-3')

    #   64 -> 128
    x = Convolution2DTranspose(
        64, kernel_size=2, strides=2, padding='same', name='R_conv3-1')(x)
    x = BatchNormalization(name='R_conv3-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1) ([x, L2]), 64, (3, 3), strides=1, name='R_conv3-2')
    x = conv_block(x, 64, (3, 3), strides=1, name='R_conv3-3')

    #   128 -> 256
    x = Convolution2DTranspose(
        32, kernel_size=2, strides=2, padding='same', name='R_conv4-1')(x)
    x = BatchNormalization(name='R_conv4-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1) ([x, L1]), 32, (3, 3), strides=1, name='R_conv4-2')
    x = conv_block(x, 32, (3, 3), strides=1, name='R_conv4-3')

    final = Conv2D(n_classes, (1, 1), name='final_out')(x)

    # final = Activation('softmax', name='softmax_1')(final)

    return final
