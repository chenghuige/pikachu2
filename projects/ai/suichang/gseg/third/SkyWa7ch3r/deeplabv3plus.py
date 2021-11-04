import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
import numpy as np


def conv_layer(inputs, filters, kernel, strides, padding, activation=True, name=None):
    conv = keras.layers.Conv2D(
        filters, kernel, strides=strides, padding=padding, name=name)(inputs)
    if name is not None:
        batch_name = name+'_batch_norm'
        relu_name = name+'_relu'
    else:
        batch_name = None
        relu_name = None
    norm = keras.layers.BatchNormalization(name=batch_name)(conv)
    if activation:
        relu = keras.layers.Activation('relu', name=relu_name)(norm)
        return relu
    else:
        return norm


def sep_layer(inputs, filters, strides, activation=True, name=None):
    sep_conv = keras.layers.SeparableConv2D(
        filters, 3, strides=strides, padding='same', name=name)(inputs)
    if name is not None:
        batch_name = name+'_batch_norm'
        relu_name = name+'_relu'
    else:
        batch_name = None
        relu_name = None
    norm = keras.layers.BatchNormalization(name=batch_name)(sep_conv)
    if activation:
        relu = keras.layers.Activation('relu', name=relu_name)(norm)
        return relu
    else:
        return norm


def entry_block(inputs):

    # First convolution
    conv1 = conv_layer(inputs, 32, 3, 2, 'same', name='entry_conv1')
    # Second Convolution
    conv2 = conv_layer(conv1, 64, 3, 1, 'same', name='entry_conv2')

    # First Separable Convolution
    sep_conv1 = sep_layer(conv2, 128, 1, name='entry_sep_conv1')
    # Second Separable Convolution
    sep_conv2 = sep_layer(sep_conv1, 128, 1, name='entry_sep_conv2')
    # Instead of a max pooling layer, we do a another Third SeparableConv, but with stride 2
    sep_conv3 = sep_layer(
        sep_conv2, 128, 2, activation=False, name='entry_sep_conv3')

    # Third Convolution (Done in parallel to SeparableConv2D Layers above)
    conv3 = conv_layer(conv2, 128, 1, 2, 'same',
                       activation=False, name='entry_skip_conv1')

    # Do a skip connection
    add1 = keras.layers.Add(name='entry_skip1')([sep_conv3, conv3])
    relu_add1 = keras.layers.Activation('relu', name='entry_relu_add1')(add1)

    # Fourth Separable Convolution
    sep_conv4 = sep_layer(relu_add1, 256, 1, name='entry_sep_conv4')
    # Fifth Separable Convolution
    sep_conv5 = sep_layer(sep_conv4, 256, 1, name='entry_sep_conv5')
    # Sixth Separable Convolution
    sep_conv6 = sep_layer(
        sep_conv5, 256, 2, activation=False, name='entry_sep_conv6')

    # Fourth Convolution (Done in parallel to SeparableConv2D Layers above)
    conv4 = conv_layer(add1, 256, 1, 2, 'same',
                       activation=False, name='entry_skip_conv2')

    # Do a skip connection
    add2 = keras.layers.Add(name='entry_skip2')([sep_conv6, conv4])
    relu_add2 = keras.layers.Activation('relu', name='entry_relu_add2')(add2)

    # Seventh Separable Convolution
    sep_conv7 = sep_layer(relu_add2, 728, 1, name='entry_sep_conv7')
    # Eighth Separable Convolution
    sep_conv8 = sep_layer(sep_conv7, 728, 1, name='entry_sep_conv8')
    # Ninth Separable Convolution
    sep_conv9 = sep_layer(
        sep_conv8, 728, 2, activation=False, name='entry_sep_conv9')

    # Fifth Convolution
    conv5 = conv_layer(add2, 728, 1, 2, 'same',
                       activation=False, name='entry_skip_conv3')

    # Do a skip connection
    add3 = keras.layers.Add(name='entry_skip3')([sep_conv9, conv5])

    # That's the entry block finished, return the last add
    return add3, relu_add2


def middle_block(inputs, blockname):
    # Activate the input
    relu_add = keras.layers.Activation(
        'relu', name=blockname+'_relu_add')(inputs)
    # Do 3 Separable Convolution Layers
    sep_conv1 = sep_layer(relu_add, 728, 1, name=blockname+'_sep_conv1')
    sep_conv2 = sep_layer(sep_conv1, 728, 1, name=blockname+'_sep_conv2')
    sep_conv3 = sep_layer(sep_conv2, 728, 1, name=blockname+'_sep_conv3')
    # Do a skip connection
    add = keras.layers.Add(name=blockname+'_skip')([inputs, sep_conv3])
    # Return the skip
    return add


def exit_block(inputs):
    # Exit Record
    sep_conv1 = sep_layer(inputs, 728, 1, name='exit_sep_conv1')
    sep_conv2 = sep_layer(sep_conv1, 1024, 1, name='exit_sep_conv2')
    sep_conv3 = sep_layer(sep_conv2, 1024, 2,
                          name='exit_sep_conv3', activation=False)
    # Skip Convolution
    conv1 = conv_layer(inputs, 1024, 1, 2, 'same',
                       activation=False, name='exit_skip_conv1')
    # Skip Connection
    add1 = keras.layers.Add(name='exit_skip1')([sep_conv3, conv1])
    relu1 = keras.layers.Activation('relu', name='exit_relu_add1')(add1)
    # Exiting Separable Convolutions
    sep_conv4 = sep_layer(relu1, 1536, 1, name='exit_sep_conv4')
    sep_conv5 = sep_layer(sep_conv4, 1536, 1, name='exit_sep_conv5')
    sep_conv6 = sep_layer(sep_conv5, 2048, 1, name='exit_sep_conv6')
    # Return and finish Xception
    return sep_conv6


def ASPP(inputs, depthwise=False, output_stride=16):
    """
    Following Code from Here:
    https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0/blob/master/deeplab.py

    For the ASPP Module, will end up having two options of with depthwise and without.
    """
    dilation_rates = np.array([(6, 6), (12, 12), (18, 18)])
    if output_stride == 8:
        dilation_rates = dilation_rates*2
    shape = list(inputs.shape)

    # Image pooling
    pool = keras.layers.AveragePooling2D(pool_size=(
        shape[1], shape[2]), name="ASPP_Ave_Pool")(inputs)
    conv1 = keras.layers.Conv2D(
        256, 1, strides=1, padding='same', use_bias=False, name='ASPP_conv1')(pool)
    norm1 = keras.layers.BatchNormalization(
        name='ASPP_conv1_batch_norm')(conv1)
    relu1 = keras.layers.Activation('relu', name='ASPP_conv1_relu')(norm1)
    upsampling = keras.layers.UpSampling2D(
        size=(shape[1], shape[2]), interpolation='bilinear')(relu1)

    # 1x1 Convolution
    conv1x1 = keras.layers.Conv2D(
        256, 1, strides=1, padding='same', use_bias=False, name='ASPP_conv1x1')(inputs)
    norm1x1 = keras.layers.BatchNormalization(
        name='ASPP_conv1x1_batch_norm')(conv1x1)
    relu1x1 = keras.layers.Activation(
        'relu', name='ASPP_conv1x1_relu')(norm1x1)

    # The Dilated Convolutions
    if depthwise:
        conv3x3_d6 = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[0], name="ASPP_sep_conv3x3_d6")(inputs)
    else:
        conv3x3_d6 = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[0], name="ASPP_conv3x3_d6")(inputs)
    norm3x3_d6 = keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d6_batch_norm")(conv3x3_d6)
    relu3x3_d6 = keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d6_relu")(norm3x3_d6)
    if depthwise:
        conv3x3_d12 = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[1], name="ASPP_sep_conv3x3_d12")(inputs)
    else:
        conv3x3_d12 = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[1], name="ASPP_conv3x3_d12")(inputs)
    norm3x3_d12 = keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d12_batch_norm")(conv3x3_d12)
    relu3x3_d12 = keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d12_relu")(norm3x3_d12)
    if depthwise:
        conv3x3_d18 = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[2], name="ASPP_sep_conv3x3_d18")(inputs)
    else:
        conv3x3_d18 = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[2], name="ASPP_conv3x3_d18")(inputs)
    norm3x3_d18 = keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d18_batch_norm")(conv3x3_d18)
    relu3x3_d18 = keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d18_relu")(norm3x3_d18)

    # Concatenate all the above layers
    concat = keras.layers.Concatenate(name='ASPP_concatenate')(
        [upsampling, relu1x1, relu3x3_d6, relu3x3_d12, relu3x3_d18])

    # Do the final convolution
    conv2 = keras.layers.Conv2D(
        256, 1, strides=1, padding='same', use_bias=False, name='ASPP_project_conv')(concat)
    norm2 = keras.layers.BatchNormalization(
        name='ASPP_project_conv_batch_norm')(conv2)
    relu2 = keras.layers.Activation(
        'relu', name='ASPP_project_conv_relu')(norm2)

    # Return the result
    return relu2


def model(input_size=(1024, 1024, 3), num_classes=20, depthwise=False, output_stride=16, backbone='xception'):
    if backbone == 'modified_xception':
        # Input Layer
        inputs = keras.Input(input_size, name='xception_input')
        # Do the entry block, also returns the low layer for ASPP and Decoder
        xception, low_layer = entry_block(inputs)
        # Now do the middle block
        for i in range(16):
            blockname = 'middle_block{}'.format(i)
            xception = middle_block(xception, blockname)
        # Activate the last add
        xception = keras.layers.Activation(
            'relu', name='exit_block_relu_add')(xception)
        # Do the exit block
        xception = exit_block(xception)
    elif backbone == 'xception':
        xception = applications.Xception(
            weights='imagenet', include_top=False, input_shape=input_size, classes=20)
        for layer in xception.layers:
            layer.trainable = False
        inputs = xception.layers[0].output
        low_layer = xception.get_layer('add_1').output
        low_layer.trainable = True
        previous = xception.layers[-1].output
        previous.trainable = True
    elif backbone == 'mobilenetv2':
        mobilenet = applications.MobileNetV2(
            weights='imagenet', include_top=False, input_shape=input_size, classes=20)
        inputs = mobilenet.layers[0].output
        low_layer = mobilenet.get_layer('block_3_depthwise').output
        previous = mobilenet.layers[-1].output
    # ASPP
    aspp = ASPP(previous, depthwise=depthwise, output_stride=output_stride)
    aspp_up = keras.layers.UpSampling2D(
        size=(4, 4), interpolation='bilinear', name='decoder_ASPP_upsample')(aspp)
    # Decoder Begins Here
    conv1x1 = keras.layers.Conv2D(
        48, 1, strides=1, padding='same', use_bias=False, name="decoder_conv1x1")(low_layer)
    norm1x1 = keras.layers.BatchNormalization(
        name='decoder_conv1x1_batch_norm')(conv1x1)
    relu1x1 = keras.layers.Activation(
        'relu', name='decoder_conv1x1_relu')(norm1x1)

    # Concatenate ASPP and the 1x1 Convolution
    decode_concat = keras.layers.Concatenate(
        name='decoder_concat')([aspp_up, relu1x1])

    # Do some Convolutions
    if depthwise:
        conv1_decoder = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', name='decoder_conv1')(decode_concat)
    else:
        conv1_decoder = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', name='decoder_conv1')(decode_concat)
    norm1_decoder = keras.layers.BatchNormalization(
        name='decoder_conv1_batch_norm')(conv1_decoder)
    relu1_decoder = keras.layers.Activation(
        'relu', name='decoder_conv1_relu')(norm1_decoder)

    if depthwise:
        conv2_decoder = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', name='decoder_conv2')(relu1_decoder)
    else:
        conv2_decoder = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', name='decoder_conv2')(relu1_decoder)
    norm2_decoder = keras.layers.BatchNormalization(
        name='decoder_conv2_batch_norm')(conv2_decoder)
    relu2_decoder = keras.layers.Activation(
        'relu', name='decoder_conv2_relu')(norm2_decoder)

    # Do classification 1x1 layer
    classification = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), strides=(
        1, 1), activation='softmax', name='Classification', dtype=tf.float32)(relu2_decoder)
    classification_up = keras.layers.UpSampling2D(size=(
        8, 8), interpolation='bilinear', name="Classificaton_Upsample", dtype=tf.float32)(classification)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=classification_up)
    # Return the model
    return model
