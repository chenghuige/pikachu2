import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def get_vgg16_weights(input_shape):
    """
    :param input_shape: The input shape for vgg16, should be same as the segnet model we are doing.

    :returns weights: The weights from convolution layers from the VGG16 model
    """
    #Get the VGG16 Model from keras
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    #Initialize the weights list
    weights = []
    #For every layer
    for layer in vgg16.layers:
        #If its a convolution layer
        if 'Conv2D' in str(layer):
            #Get the weights and biases!
            weights.append(layer.get_weights())
    #Return the list of weights and biases from VGG16 ImageNet
    return weights 

def get_conv_layers_for_vgg16(model):
    """
    Gets the indexes of the convolution layers we wish to apply the pretrained weights to.

    :param model: A segnet model that has been made already

    :returns indexes: A list of indexes of convolutional layers
    """
    layers = model.layers
    i = 0
    indexes = []
    while "Unpooling" not in str(layers[i]):
        if "Conv2D" in str(layers[i]):
            indexes.append(i)
        i+=1
    return indexes

def conv_layer(input_tensor, channels, weight_decay=0.0005):
    """
    A convolution block which contains a convolution.
    Then a Batch Normalization.
    Then a ReLU activation.

    :param input: A Tensor, which is the output of the previous layer
    :param filters: The number of channels it should output
    :param training: Defaults as True, mainly for Batch, if validation
    """
    #Do the Convolution
    conv = keras.layers.Conv2D(channels, 
                               3, 
                               padding='same', 
                               kernel_regularizer=keras.regularizers.l2(weight_decay), 
                               bias_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
    #Do a Batch Normalization
    norm = keras.layers.BatchNormalization()(conv)
    #Then a ReLU activation
    relu = keras.layers.Activation('relu')(norm)
    #Return the result of the block
    return relu

def model(num_classes=19, input_size=(1024,1024,3), pool=(2,2)):
    '''
    In the original github repo here, https://github.com/toimcio/SegNet-tensorflow/blob/master/SegNet.py
    Here I will mix keras and tf.nn layers to produce the same model. Using parts of the github repo above
    to fill in with tf.nn when necessary. This makes the other convolutions a little easier to read I reckon.

    :param input_size: By default (1024, 1024,3)

    :returns model: A Bayesian Segmentation Network as per the paper https://arxiv.org/abs/1511.02680
    '''
    ###-----INPUTS-----###

    #Setup the inputs layer
    inputs = keras.Input(shape=input_size)
    #Do the Local Response Normalization
    lrn = tf.nn.local_response_normalization(inputs, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)

    ###-----ENCODER-----###
    #This is essentially VGG16 with some minor additions

    #First Block of Encoder, 2 Convolutions and then a Max Pooling Layer (which takes the indices)
    conv1 = conv_layer(lrn, 64)
    conv2 = conv_layer(conv1, 64)
    pool1, pool1_indices = MaxPoolingWithArgmax2D(pool)(conv2)

    #Second Block of Encoder, same as the first, but 128 channels
    conv3 = conv_layer(pool1, 128)
    conv4 = conv_layer(conv3, 128)
    pool2, pool2_indices = MaxPoolingWithArgmax2D(pool)(conv4)

    # Third Block of Encoder, 3 convolutions with 256 channels and a max pooling
    conv5 = conv_layer(pool2, 256)
    conv6 = conv_layer(conv5, 256)
    conv7 = conv_layer(conv6, 256)
    pool3, pool3_indices = MaxPoolingWithArgmax2D(pool)(conv7)

    #Fourth Block of Encoder, A Dropout Layer, 3 Convolutions with 512 channels, then a pooling layer
    drop1 = keras.layers.Dropout(0.5)(pool3)
    conv8 = conv_layer(drop1, 512)
    conv9 = conv_layer(conv8, 512)
    conv10 = conv_layer(conv9, 512)
    pool4, pool4_indices = MaxPoolingWithArgmax2D(pool)(conv10)

    #Fifth Block of Encoder, Same as Foruth
    drop2 = keras.layers.Dropout(0.5)(pool4)
    conv11 = conv_layer(drop2, 512)
    conv12 = conv_layer(conv11, 512)
    conv13 = conv_layer(conv12, 512)
    pool5, pool5_indices = MaxPoolingWithArgmax2D(pool)(conv13)

    ###-----DECODER-----###

    #First Block of Decoder, A Dropout, Upsampling, then 3 Convolutions of 512 channels
    drop3 = keras.layers.Dropout(0.5)(pool5)
    up1 = MaxUnpooling2D(pool)([drop3, pool5_indices])
    deconv1 = conv_layer(up1, 512)
    deconv2 = conv_layer(deconv1, 512)
    deconv3 = conv_layer(deconv2, 512)

    #Second Block of Decoder, Same as the first, except last convolution is 256 channels
    drop4 = keras.layers.Dropout(0.5)(deconv3)
    up2 = MaxUnpooling2D(pool)([drop4, pool4_indices])
    deconv4 = conv_layer(up2, 512)
    deconv5 = conv_layer(deconv4, 512)
    deconv6 = conv_layer(deconv5, 256)

    #Third Block of Decoder, Same as the second, except last convolution is 128 channels
    drop5 = keras.layers.Dropout(0.5)(deconv6)
    up3 = MaxUnpooling2D(pool)([drop5, pool3_indices])
    deconv7 = conv_layer(up3, 256)
    deconv8 = conv_layer(deconv7, 256)
    deconv9 = conv_layer(deconv8, 128)

    #Fourth Block of Decoder, Same as before 
    drop6 = keras.layers.Dropout(0.5)(deconv9)
    up4 = MaxUnpooling2D(pool)([drop6, pool2_indices])
    deconv10 = conv_layer(up4, 128)
    deconv11 = conv_layer(deconv10, 64)

    #Fifth Block of Decoder
    up5 = MaxUnpooling2D(pool)([deconv11, pool1_indices])
    deconv12 = conv_layer(up5, 64)
    deconv13 = conv_layer(deconv12, 64)

    #Classifier
    outputs = keras.layers.Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation = 'softmax', dtype=tf.float32)(deconv13)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Bayes_Segnet")

    #Now apply the weights from the VGG16 ImageNet Model
    weights = get_vgg16_weights(input_size)
    convolution_indexes = get_conv_layers_for_vgg16(model)

    #Now zip them together, and use set weights to put the weights in
    for weight, index in zip(weights, convolution_indexes):
        model.layers[index].set_weights(weight)

    return model












