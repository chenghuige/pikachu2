
"""A human friendly ResNet implementation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers


def residual_block(filters=64, downsample=False, kernel_size=(3, 3),
                   padding='same', activation='relu'):
    """Building block for shallow ResNet."""

    # Note down sampling is performed by the first conv layer. Batch
    # normalization (BN) adopted right after each convolution and before
    # activation.
    strides = (2, 2) if downsample else (1, 1)
    block_layers = [layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  activation=None),
                    layers.BatchNormalization(),
                    layers.Activation(activation)]

    # Then the second conv layer without down sampling.
    block_layers.extend([layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=(1, 1),
                                       padding=padding,
                                       activation=None),
                         layers.BatchNormalization()])

    # Down sample the input if the feature maps are down sampled.
    if downsample:
        matching_layers = [layers.Conv2D(filters=filters,
                                         kernel_size=(1, 1),
                                         strides=strides,
                                         padding=padding,
                                         activation=None),
                           layers.BatchNormalization()]

    def forward(inputs):
        x = inputs
        for layer in block_layers:
            x = layer(x)

        # Match the feature map size and channels.
        if downsample:
            for layer in matching_layers:
                inputs = layer(inputs)

        # Time for the famous shortcut connection.
        x = layers.Add()([x, inputs])

        # Finally, output of the block.
        x = layers.Activation(activation)(x)

        return x

    return forward


def residual_blocks(num_blocks=2, filters=64, downsample=False, activation='relu'):
    block_layers = [
        residual_block(filters, downsample=downsample, activation=activation)]
    block_layers.extend(
        [residual_block(filters, downsample=False, activation=activation)
         for _ in range(num_blocks - 1)])

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def bottleneck_block(filters=64, expantion=1, kernel_size=(3, 3), downsample=False, padding='same', activation='relu'):
    """Building block for deeper ResNet."""

    # Bottleneck block could be expanded. Get the expanded size.
    filters *= expantion

    # Note down sampling is performed by the first conv layer. Batch
    # normalization (BN) adopted right after each convolution and before
    # activation.
    strides = (2, 2) if downsample else (1, 1)
    block_layers = [layers.Conv2D(filters=filters,
                                  kernel_size=(1, 1),
                                  strides=strides,
                                  padding=padding,
                                  activation=None),
                    layers.BatchNormalization(),
                    layers.Activation(activation)]

    # Then the second conv layer without down sampling.
    block_layers.extend([layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=(1, 1),
                                       padding=padding,
                                       activation=None),
                         layers.BatchNormalization(),
                         layers.Activation(activation)])

    # Then the third conv layer, also without down sampling. This layer also
    # has 4 times more filters.
    block_layers.extend([layers.Conv2D(filters=filters * 4,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding=padding,
                                       activation=None),
                         layers.BatchNormalization()])

    # Time for the famous shortcut connection. Down sample the input if the
    # feature maps are down sampled.
    matching_layers = [layers.Conv2D(filters=filters * 4,
                                     kernel_size=(1, 1),
                                     strides=strides,
                                     padding=padding,
                                     activation=None),
                       layers.BatchNormalization()]

    def forward(inputs):
        x = inputs
        for layer in block_layers:
            x = layer(x)

        # Match the feature map size and channels.
        for layer in matching_layers:
            inputs = layer(inputs)

        x = layers.Add()([x, inputs])

        # Finally, output of the block.
        x = layers.Activation(activation)(x)

        return x

    return forward


def bottleneck_blocks(num_blocks=3, filters=64, downsample=False, activation='relu'):
    block_layers = [
        bottleneck_block(filters, downsample=downsample, activation=activation)]
    block_layers.extend([
        bottleneck_block(filters, downsample=False, activation=activation)
        for _ in range(num_blocks - 1)])

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def rsn_stem(filters=64, kernel_size=(7, 7), strides=(2, 2), pool_size=(3, 3)):
    block_layers = [layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same'),
                    layers.MaxPooling2D(pool_size=pool_size,
                                        strides=strides,
                                        padding='same')]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def rsn_head(output_size):
    block_layers = [layers.GlobalAveragePooling2D(), layers.Dense(output_size)]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def make_resnet(block_size_list, bottleneck=False, input_shape=(256, 256, 3), output_size=1000, name='ResNet'):
    # Make sure the network setup is valid.
    assert len(block_size_list) == 4, \
        "Blocks size should be a list of 4 int numbers."

    block_type = bottleneck_blocks if bottleneck else residual_blocks
    filters_list = [64, 128, 256, 512]

    # Construct a ResNet model
    inputs = keras.Input(shape=input_shape, name="input_image_tensor")

    # Conv1
    x = rsn_stem(filters=64, kernel_size=(7, 7), strides=(2, 2))(inputs)

    # Conv2 -> Conv5
    body_layers = []
    for stage, num_blocks, filters in zip(range(4), block_size_list, filters_list):
        downsample = True if stage > 0 else False
        body_layers.append(block_type(num_blocks=num_blocks, filters=filters,
                                      downsample=downsample,
                                      activation='relu'))
    for layer in body_layers:
        x = layer(x)

    # Output head.
    outputs = rsn_head(output_size)(x)

    # Assemble the model.
    model = Model(inputs, outputs, name=name)

    return model


class ResidualBlock(layers.Layer):
    def __init__(self, filters=64, downsample=False, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.filters = filters
        self.activation_fun = activation
        self.downsample = downsample

    def build(self, input_shape):
        """
        The build() function makes lazily creating weights possible.
        """

        # First conv layer with/without down sampling.
        strides = (2, 2) if self.downsample else (1, 1)
        self.conv2d_1 = layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding='same',
                                      activation=None)

        # Second Conv layer without down sampling.
        self.conv2d_2 = layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None)

        # Batch normalization layer.
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()

        # Activation layer.
        self.Activation = layers.Activation

        # Shortcut connection.
        self.shortcut = layers.Add()

        # In case the inputs are down sampled.
        if self.downsample:
            self.downsample_inputs = layers.Conv2D(filters=self.filters,
                                                   kernel_size=(1, 1),
                                                   strides=strides,
                                                   padding='same',
                                                   activation=None)
            self.batch_norm_3 = layers.BatchNormalization()

        self.built = True

    def call(self, inputs):
        # First conv.
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.Activation(self.activation_fun)(x)

        # Second conv.
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)

        # Shortcut.
        if self.downsample:
            inputs = self.downsample_inputs(inputs)
            inputs = self.batch_norm_3(inputs)
        x = self.shortcut([x, inputs])

        # Output.
        x = self.Activation(self.activation_fun)(x)

        return x

    def get_config(self):
        """
        Override this function so keras can serialize the layer.
        """
        config = super(ResidualBlock, self).get_config()
        config.update({"filters": self.filters,
                       "downsample": self.downsample,
                       "activation": self.activation_fun})
        return config


class ResidualBlocks(layers.Layer):
    """A bunch of Residual Blocks. Down sampling is only performed by the first
    block if required."""

    def __init__(self, num_blocks=2, filters=64, downsample=False,
                 activation='relu', **kwargs):
        super(ResidualBlocks, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.filters = filters
        self.downsample = downsample
        self.activation = activation

    def build(self, input_shape):
        self.block_1 = ResidualBlock(self.filters, self.downsample,
                                     self.activation)
        self.blocks = [ResidualBlock(self.filters, False, self.activation)
                       for _ in range(self.num_blocks - 1)]

        self.built = True

    def call(self, inputs):
        x = self.block_1(inputs)
        for block in self.blocks:
            x = block(x)

        return x

    def get_config(self):
        config = super(ResidualBlocks, self).get_config()
        config.update({"num_blocks": self.num_blocks,
                       "filters": self.filters,
                       "downsample": self.downsample,
                       "activation": self.activation})
        return config


class BottleneckBlock(layers.Layer):

    def __init__(self, filters=64, downsample=False, activation='relu', **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)

        self.filters = filters
        self.downsample = downsample
        self.activation_fun = activation

    def build(self, input_shape):
        # First conv layer with/without down sampling.
        strides = (2, 2) if self.downsample else (1, 1)
        self.conv2d_1 = layers.Conv2D(filters=self.filters,
                                      kernel_size=(1, 1),
                                      strides=strides,
                                      padding='same',
                                      activation=None)

        # Second Conv layer without down sampling.
        self.conv2d_2 = layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None)

        # Third Conv layer without down sampling.
        self.conv2d_3 = layers.Conv2D(filters=self.filters*4,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None)

        # Batch normalization layer.
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.batch_norm_3 = layers.BatchNormalization()
        self.batch_norm_4 = layers.BatchNormalization()

        # Activation layer.
        self.Activation = layers.Activation

        # Shortcut connection.
        self.shortcut = layers.Add()

        # In case the inputs are down sampled.
        self.match_inputs = layers.Conv2D(filters=self.filters*4,
                                          kernel_size=(1, 1),
                                          strides=strides,
                                          padding='same',
                                          activation=None)

        self.built = True

    def call(self, inputs):
        # First conv.
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.Activation(self.activation_fun)(x)

        # Second conv.
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        x = self.Activation(self.activation_fun)(x)

        # Third conv.
        x = self.conv2d_3(x)
        x = self.batch_norm_3(x)
        x = self.Activation(self.activation_fun)(x)

        # Shortcut.
        inputs = self.match_inputs(inputs)
        inputs = self.batch_norm_4(inputs)
        x = self.shortcut([x, inputs])

        # Output.
        x = self.Activation(self.activation_fun)(x)

        return x

    def get_config(self):
        config = super(BottleneckBlock, self).get_config()
        config.update({"filters": self.filters,
                       "downsample": self.downsample,
                       "activation": self.activation_fun})

        return config


class BottleneckBlocks(layers.Layer):
    """A bunch of Bottleneck Blocks. Down sampling is only performed by the
    first block if required."""

    def __init__(self, num_blocks=3, filters=64, downsample=False,
                 activation='relu', **kwargs):
        super(BottleneckBlocks, self).__init__(**kwargs)

        self.num_blocks = num_blocks
        self.filters = filters
        self.downsample = downsample
        self.activation = activation

    def build(self, input_shape):
        self.block_1 = BottleneckBlock(self.filters, self.downsample,
                                       self.activation)
        self.blocks = [BottleneckBlock(self.filters, False, self.activation)
                       for _ in range(self.num_blocks - 1)]

        self.built = True

    def call(self, inputs):
        x = self.block_1(inputs)
        for block in self.blocks:
            x = block(x)

        return x

    def get_config(self):
        config = super(BottleneckBlocks, self).get_config()
        config.update({"num_blocks": self.num_blocks,
                       "filters": self.filters,
                       "downsample": self.downsample,
                       "activation": self.activation})

        return config


class RSNHead(layers.Layer):
    def __init__(self, filters=64, kernel_size=(7, 7), strides=(2, 2),
                 pool_size=(3, 3), **kwargs):
        super(RSNHead, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.maxpool2d = layers.MaxPooling2D(pool_size=self.pool_size,
                                             strides=self.strides,
                                             padding='same')

        self.built = True

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool2d(x)

        return x

    def get_config(self):
        config = super(RSNHead, self).get_config()
        config.update({"filters": self.filters,
                       "kernel_size": self.kernel_size,
                       "strides": self.strides,
                       "pool_size": self.pool_size})
        return config


class RSNTail(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super(RSNTail, self).__init__(**kwargs)
        self.output_size = output_size

    def build(self, input_shape):
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(self.output_size)

        self.built = True

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.fc(x)

        return x

    def get_config(self):
        config = super(RSNTail, self).get_config()
        config.update({"output_size": self.output_size})

        return config


class ResNet(Model):
    def __init__(self, block_size_list, bottleneck=False, output_size=1000, **kwargs):
        # Make sure the network setup is valid.
        assert len(block_size_list) == 4, \
            "Blocks size should be a list of 4 int numbers."

        super(ResNet, self).__init__(**kwargs)
        BlockClass = BottleneckBlocks if bottleneck else ResidualBlocks
        filters_list = [64, 128, 256, 512]

        # Conv1
        self.conv_1 = RSNHead(filters=64, kernel_size=(7, 7), strides=(2, 2))

        # Conv2 -> Conv5
        self.convs = []
        for stage, num_blocks, filters in zip(range(4), block_size_list, filters_list):
            downsample = True if stage > 0 else False
            self.convs.append(BlockClass(num_blocks=num_blocks, filters=filters,
                                         downsample=downsample,
                                         activation='relu'))

        # Output
        self.tail = RSNTail(output_size)

    def call(self, inputs):
        # Conv1
        x = self.conv_1(inputs)

        # Conv2 -> Conv5
        for conv in self.convs:
            x = conv(x)

        # Output
        x = self.tail(x)

        return x

    def prepare_summary(self, input_shape):
        """Prepare the subclassed model for summary.Check out TensorFlow issue
        29132 for the original code.

        Args:
            input_shape: A shape tuple (integers), not including the batch size.
        """
        self.build((1,) + input_shape)
        self.call(keras.Input(input_shape))
