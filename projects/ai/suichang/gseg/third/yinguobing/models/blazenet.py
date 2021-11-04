"""Building blocks for BlazeFace backbone network."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def zero_channel_padding(channels_to_pad):
    """Padding the tensor channels with zero."""
    paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, channels_to_pad]])

    def forward(inputs):
        return tf.pad(inputs, paddings, "CONSTANT")

    return forward


def blaze_block(filters, strides=(1, 1)):
    block_layers = [
        layers.DepthwiseConv2D(kernel_size=(5, 5),
                               strides=strides,
                               padding='same'),
        layers.Conv2D(filters=filters,
                      kernel_size=(1, 1),
                      padding='same'),
        layers.BatchNormalization()]

    def forward(inputs):
        x = inputs
        for layer in block_layers:
            x = layer(x)

        # Optional layers
        inputs = layers.MaxPool2D(pool_size=(5, 5),
                                  strides=strides,
                                  padding='same')(inputs)
        channels_to_pad = filters - inputs.shape.as_list()[3]

        if channels_to_pad > 0:
            inputs = zero_channel_padding(channels_to_pad)(inputs)
        elif channels_to_pad < 0:
            inputs = layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   padding='same')(inputs)
            inputs = layers.BatchNormalization()(inputs)

        # Shortcut connection.
        shortcut = layers.Add()([x, inputs])
        return layers.Activation("relu")(shortcut)

    return forward


def double_blaze_block(filters_1, filters_2, strides=(1, 1)):
    block_layers = [
        layers.DepthwiseConv2D(kernel_size=(5, 5),
                               strides=strides,
                               padding='same'),
        layers.Conv2D(filters=filters_1,
                      kernel_size=(1, 1),
                      padding='same'),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.DepthwiseConv2D(kernel_size=(5, 5),
                               padding='same'),
        layers.Conv2D(filters=filters_2,
                      kernel_size=(1, 1),
                      padding='same'),
        layers.BatchNormalization()]

    def forward(inputs):
        x = inputs
        for layer in block_layers:
            x = layer(x)

        # Optional layers
        inputs = layers.MaxPool2D(pool_size=(5, 5),
                                  strides=strides,
                                  padding='same')(inputs)
        channels_to_pad = filters_2 - inputs.shape.as_list()[3]

        if channels_to_pad > 0:
            inputs = zero_channel_padding(channels_to_pad)(inputs)
        elif channels_to_pad < 0:
            inputs = layers.Conv2D(filters_2, (1, 1))(inputs)
            inputs = layers.BatchNormalization()(inputs)

        # Shortcut connection.
        shortcut = layers.Add()([x, inputs])
        return layers.Activation("relu")(shortcut)

    return forward
