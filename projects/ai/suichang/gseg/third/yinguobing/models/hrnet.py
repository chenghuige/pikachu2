"""A human friendly implimentation of High-Resolution Net."""

from itertools import chain

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers

from .resnet import (BottleneckBlock, ResidualBlock, bottleneck_block,
                           residual_block)


def hrn_1st_stage(filters=64, activation='relu'):
    block_layers = [bottleneck_block(filters=filters),
                    bottleneck_block(filters=filters),
                    bottleneck_block(filters=filters),
                    bottleneck_block(filters=filters),
                    layers.Conv2D(filters=filters,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding='same'),
                    layers.BatchNormalization()]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def hrn_block(filters=64, activation='relu'):
    block_layers = [residual_block(filters, activation=activation),
                    residual_block(filters, activation=activation),
                    residual_block(filters, activation=activation),
                    residual_block(filters, activation=activation)]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def hrn_blocks(repeat=1, filters=64, activation='relu'):
    block_layers = [hrn_block(filters, activation=activation)
                    for _ in range(repeat)]

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def fusion_layer(filters, upsample=False, activation='relu'):
    block_layers = []
    if upsample:
        block_layers.extend([layers.Conv2D(filters=filters,
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding='same'),
                             layers.UpSampling2D(size=(2, 2),
                                                 interpolation='bilinear')])
    else:
        block_layers.append(layers.Conv2D(filters=filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same'))

    block_layers.extend([layers.BatchNormalization(),
                         layers.Activation(activation)])

    def forward(inputs):
        for layer in block_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def fusion_block(filters, branches_in, branches_out, activation='relu'):
    """A fusion block will fuse multi-resolution inputs.

    A typical fusion block looks like a square box with cells. For example at
    stage 3, the fusion block consists 12 cells. Each cell represents a fusion
    layer. Every cell whose row < column is a down sampling cell, whose row ==
    column is a identity cell, and the rest are up sampling cells.

             B1         B2         B3         B4
        |----------|----------|----------|----------|
    B1  | identity |    ->    |    ->    |    ->    |
        |----------|----------|----------|----------|
    B2  |    <-    | identity |    ->    |    ->    |
        |----------|----------|----------|----------|
    B3  |    <-    |    <-    | identity |    ->    |
        |----------|----------|----------|----------|
    """
    # Construct the fusion layers.
    _fusion_grid = []

    rows = branches_in
    columns = branches_out

    for row in range(rows):
        _fusion_layers = []
        for column in range(columns):
            if column == row:
                _fusion_layers.append(tf.identity)
            elif column > row:
                # Down sampling.
                _fusion_layers.append(fusion_layer(filters * pow(2, column),
                                                   False, activation))
            else:
                # Up sampling.
                _fusion_layers.append(fusion_layer(filters * pow(2, column),
                                                   True, activation))

        _fusion_grid.append(_fusion_layers)

    if len(_fusion_grid) > 1:
        _add_layers_group = [layers.Add() for _ in range(branches_out)]

    def forward(inputs):
        rows = len(_fusion_grid)
        columns = len(_fusion_grid[0])

        # Every cell in the fusion grid has an output value.
        fusion_values = [[None for _ in range(columns)] for _ in range(rows)]

        for row in range(rows):
            # The down sampling operation excutes from left to right.
            for column in range(columns):
                # The input will be different for different cells.
                if column < row:
                    # Skip all up samping cells.
                    continue
                elif column == row:
                    # The input is the branch output.
                    x = inputs[row]
                elif column > row:
                    # Down sampling, the input is the fusion value of the left cell.
                    x = fusion_values[row][column - 1]

                fusion_values[row][column] = _fusion_grid[row][column](x)

            # The upsampling operation excutes in the opposite direction.
            for column in reversed(range(columns)):
                if column >= row:
                    # Skip all down samping and identity cells.
                    continue

                x = fusion_values[row][column + 1]
                fusion_values[row][column] = _fusion_grid[row][column](x)

        # The fused value for each branch.
        if rows == 1:
            outputs = [fusion_values[0][0], fusion_values[0][1]]
        else:
            outputs = []
            fusion_values = [list(v) for v in zip(*fusion_values)]

            for index, values in enumerate(fusion_values):
                outputs.append(_add_layers_group[index](values))

        return outputs

    return forward


def hrnet_body(filters=64):
    # Stage 1
    s1_b1 = hrn_1st_stage(filters)
    s1_fusion = fusion_block(filters, branches_in=1, branches_out=2)

    # Stage 2
    s2_b1 = hrn_block(filters)
    s2_b2 = hrn_block(filters*2)
    s2_fusion = fusion_block(filters, branches_in=2, branches_out=3)

    # Stage 3
    s3_b1 = hrn_blocks(4, filters)
    s3_b2 = hrn_blocks(4, filters*2)
    s3_b3 = hrn_blocks(4, filters*4)
    s3_fusion = fusion_block(filters, branches_in=3, branches_out=4)

    # Stage 4
    s4_b1 = hrn_blocks(3, filters)
    s4_b2 = hrn_blocks(3, filters*2)
    s4_b3 = hrn_blocks(3, filters*4)
    s4_b4 = hrn_blocks(3, filters*8)

    def forward(inputs):
        # Stage 1
        x = s1_b1(inputs)
        x = s1_fusion([x])

        # Stage 2
        x_1 = s2_b1(x[0])
        x_2 = s2_b2(x[1])
        x = s2_fusion([x_1, x_2])

        # Stage 3
        x_1 = s3_b1(x[0])
        x_2 = s3_b2(x[1])
        x_3 = s3_b3(x[2])
        x = s3_fusion([x_1, x_2, x_3])

        # Stage 4
        x_1 = s4_b1(x[0])
        x_2 = s4_b2(x[1])
        x_3 = s4_b3(x[2])
        x_4 = s4_b4(x[3])

        return [x_1, x_2, x_3, x_4]

    return forward


class HRN1stStage(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, filters=64, activation='relu', **kwargs):
        super(HRN1stStage, self).__init__(**kwargs)

        self.filters = filters
        self.activation = activation

    def build(self, input_shape):
        self.bottleneck_1 = BottleneckBlock(64)
        self.bottleneck_2 = BottleneckBlock(64)
        self.bottleneck_3 = BottleneckBlock(64)
        self.bottleneck_4 = BottleneckBlock(64)
        self.conv3x3 = layers.Conv2D(filters=self.filters,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')
        self.s1_batch_norm = layers.BatchNormalization()

        self.built = True

    def call(self, inputs):
        x = self.bottleneck_1(inputs)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.conv3x3(x)
        x = self.s1_batch_norm(x)

        return x

    def get_config(self):
        config = super(HRN1stStage, self).get_config()
        config.update({"filters": self.filters, "activation": self.activation})

        return config

    def get_prunable_weights(self):
        prunable_weights = list(chain(*[
            self.bottleneck_1.get_prunable_weights(),
            self.bottleneck_2.get_prunable_weights(),
            self.bottleneck_3.get_prunable_weights(),
            self.bottleneck_4.get_prunable_weights(),
            [getattr(self.conv3x3, 'kernel')]
        ]))

        return prunable_weights


class HRNBlock(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, filters=64, activation='relu', **kwargs):
        super(HRNBlock, self).__init__(**kwargs)

        self.filters = filters
        self.activation = activation

    def build(self, input_shape):
        # There are 4 residual blocks in each modularized block.
        self.residual_block_1 = ResidualBlock(self.filters, False,
                                              self.activation)
        self.residual_block_2 = ResidualBlock(self.filters, False,
                                              self.activation)
        self.residual_block_3 = ResidualBlock(self.filters, False,
                                              self.activation)
        self.residual_block_4 = ResidualBlock(self.filters, False,
                                              self.activation)

        self.built = True

    def call(self, inputs):
        x = self.residual_block_1(inputs)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)

        return x

    def get_config(self):
        config = super(HRNBlock, self).get_config()
        config.update({"filters": self.filters, "activation": self.activation})

        return config

    def get_prunable_weights(self):
        prunable_weights = list(chain(*[
            self.residual_block_1.get_prunable_weights(),
            self.residual_block_2.get_prunable_weights(),
            self.residual_block_3.get_prunable_weights(),
            self.residual_block_4.get_prunable_weights()]))

        return prunable_weights


class HRNBlocks(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, repeat=1, filters=64, activation='relu', **kwargs):
        super(HRNBlocks, self).__init__(**kwargs)

        self.repeat = repeat
        self.filters = filters
        self.activation = activation

    def build(self, input_shape):
        self.blocks = [HRNBlock(self.filters, self.activation)
                       for _ in range(self.repeat)]

        self.built = True

    def call(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)

        return inputs

    def get_config(self):
        config = super(HRNBlocks, self).get_config()
        config.update({"repeat": self.repeat, "filters": self.filters,
                       "activation": self.activation})

        return config

    def get_prunable_weights(self):
        prunable_weights = list(chain(*[block.get_prunable_weights()
                                        for block in self.blocks]))

        return prunable_weights


class FusionLayer(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """A fusion layer actually do two things: resize the maps, match the channels"""

    def __init__(self, filters, upsample=False, activation='relu', **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

        self.filters = filters
        self.upsample = upsample
        self.activation_fun = activation

    def build(self, input_shape):
        if self.upsample:
            self.upsample_layer = layers.UpSampling2D(size=(2, 2),
                                                      interpolation='bilinear')
            self.match_channels = layers.Conv2D(filters=self.filters,
                                                kernel_size=(1, 1),
                                                strides=(1, 1),
                                                padding='same')
        else:
            self.downsample_layer = layers.Conv2D(filters=self.filters,
                                                  kernel_size=(3, 3),
                                                  strides=(2, 2),
                                                  padding='same')

        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation(self.activation_fun)

        self.built = True

    def call(self, inputs):
        if self.upsample:
            x = self.match_channels(inputs)
            x = self.upsample_layer(x)
        else:
            x = self.downsample_layer(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x

    def get_config(self):
        config = super(FusionLayer, self).get_config()
        config.update({"filters": self.filters, "upsample": self.upsample,
                       "activation": self.activation_fun})

        return config

    def get_prunable_weights(self):
        if self.upsample:
            prunable_weights = [getattr(self.match_channels, 'kernel')]
        else:
            prunable_weights = [getattr(self.downsample_layer, 'kernel')]

        return prunable_weights


class Identity(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """A identity layer do NOT modify the tensors."""

    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return tf.identity(inputs)

    def get_config(self):
        return super(Identity, self).get_config()

    def get_prunable_weights(self):
        return []


class FusionBlock(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """A fusion block will fuse multi-resolution inputs.

    A typical fusion block looks like a square box with cells. For example at
    stage 3, the fusion block consists 12 cells. Each cell represents a fusion
    layer. Every cell whose row < column is a down sampling cell, whose row ==
    column is a identity cell, and the rest are up sampling cells.

             B1         B2         B3         B4
        |----------|----------|----------|----------|
    B1  | identity |    ->    |    ->    |    ->    |
        |----------|----------|----------|----------|
    B2  |    <-    | identity |    ->    |    ->    |
        |----------|----------|----------|----------|
    B3  |    <-    |    <-    | identity |    ->    |
        |----------|----------|----------|----------|
    """

    def __init__(self, filters, branches_in, branches_out, activation='relu', **kwargs):
        super(FusionBlock, self).__init__(**kwargs)

        self.filters = filters
        self.branches_in = branches_in
        self.branches_out = branches_out
        self.activation = activation

    def build(self, input_shape):
        # Construct the fusion layers.
        self._fusion_grid = []

        for row in range(self.branches_in):
            fusion_layers = []
            for column in range(self.branches_out):
                if column == row:
                    fusion_layers.append(Identity())
                elif column > row:
                    # Down sampling.
                    fusion_layers.append(FusionLayer(self.filters * pow(2, column),
                                                     False, self.activation))
                else:
                    # Up sampling.
                    fusion_layers.append(FusionLayer(self.filters * pow(2, column),
                                                     True, self.activation))

            self._fusion_grid.append(fusion_layers)

        if len(self._fusion_grid) > 1:
            self._add_layers_group = [layers.Add()
                                      for _ in range(self.branches_out)]

        self.built = True

    def call(self, inputs):
        """Fuse the last layer's outputs. The inputs should be a list of the last layers output tensors in order of branches."""
        rows = len(self._fusion_grid)
        columns = len(self._fusion_grid[0])

        # Every cell in the fusion grid has an output value.
        fusion_values = [[None for _ in range(columns)] for _ in range(rows)]

        for row in range(rows):
            # The down sampling operation excutes from left to right.
            for column in range(columns):
                # The input will be different for different cells.
                if column < row:
                    # Skip all up samping cells.
                    continue
                elif column == row:
                    # The input is the branch output.
                    x = inputs[row]
                elif column > row:
                    # Down sampling, the input is the fusion value of the left cell.
                    x = fusion_values[row][column - 1]
                fusion_values[row][column] = self._fusion_grid[row][column](x)

            # The upsampling operation excutes in the opposite direction.
            for column in reversed(range(columns)):
                if column >= row:
                    # Skip all down samping and identity cells.
                    continue
                x = fusion_values[row][column + 1]
                fusion_values[row][column] = self._fusion_grid[row][column](x)

        # The fused value for each branch.
        if rows == 1:
            outputs = [fusion_values[0][0], fusion_values[0][1]]
        else:
            outputs = []
            fusion_values = [list(v) for v in zip(*fusion_values)]

            for index, values in enumerate(fusion_values):
                outputs.append(self._add_layers_group[index](values))

        return outputs

    def get_config(self):
        config = super(FusionBlock, self).get_config()
        config.update({"filters": self.filters,
                       "branches_in": self.branches_in,
                       "branches_out": self.branches_out,
                       "activation":  self.activation})

        return config

    def get_prunable_weights(self):
        prunable_weights = []
        for _layers in self._fusion_grid:
            for _layer in _layers:
                prunable_weights.extend(
                    list(chain(_layer.get_prunable_weights())))

        return prunable_weights


class HRNetBody(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, filters=64, **kwargs):
        super(HRNetBody, self).__init__(**kwargs)

        self.filters = filters

    def build(self, input_shape):
        # Stage 1
        self.s1_b1_block = HRN1stStage(self.filters, name="s1_b1")

        self.s1_fusion = FusionBlock(self.filters, branches_in=1, branches_out=2,
                                     name="fusion_1")

        # Stage 2
        self.s2_b1_block = HRNBlock(self.filters, name="s2_b1")
        self.s2_b2_block = HRNBlock(self.filters*2, name="s2_b2")

        self.s2_fusion = FusionBlock(self.filters, branches_in=2, branches_out=3,
                                     name="fusion_2")

        # Stage 3
        self.s3_b1_blocks = HRNBlocks(4, self.filters, name="s3_b1")
        self.s3_b2_blocks = HRNBlocks(4, self.filters*2, name="s3_b2")
        self.s3_b3_blocks = HRNBlocks(4, self.filters*4, name="s3_b3")

        self.s3_fusion = FusionBlock(self.filters, branches_in=3, branches_out=4,
                                     name="fusion_3")

        # Stage 4
        self.s4_b1_blocks = HRNBlocks(3, self.filters, name="s4_b1")
        self.s4_b2_blocks = HRNBlocks(3, self.filters*2, name="s4_b2")
        self.s4_b3_blocks = HRNBlocks(3, self.filters*4, name="s4_b3")
        self.s4_b4_blocks = HRNBlocks(3, self.filters*8, name="s4_b4")

        self.built = True

    def call(self, inputs):
        # Stage 1
        x = self.s1_b1_block(inputs)
        x = self.s1_fusion([x])

        # Stage 2
        x_1 = self.s2_b1_block(x[0])
        x_2 = self.s2_b2_block(x[1])
        x = self.s2_fusion([x_1, x_2])

        # Stage 3
        x_1 = self.s3_b1_blocks(x[0])
        x_2 = self.s3_b2_blocks(x[1])
        x_3 = self.s3_b3_blocks(x[2])
        x = self.s3_fusion([x_1, x_2, x_3])

        # Stage 4
        x_1 = self.s4_b1_blocks(x[0])
        x_2 = self.s4_b2_blocks(x[1])
        x_3 = self.s4_b3_blocks(x[2])
        x_4 = self.s4_b4_blocks(x[3])

        return [x_1, x_2, x_3, x_4]

    def get_config(self):
        config = super(HRNetBody, self).get_config()
        config.update({"filters": self.filters})

        return config

    def get_prunable_weights(self):
        prunable_weights = list(chain(*[
            self.s1_b1_block.get_prunable_weights(),
            self.s1_fusion.get_prunable_weights(),
            self.s2_b1_block.get_prunable_weights(),
            self.s2_b2_block.get_prunable_weights(),
            self.s2_fusion.get_prunable_weights(),
            self.s3_b1_blocks.get_prunable_weights(),
            self.s3_b2_blocks.get_prunable_weights(),
            self.s3_b3_blocks.get_prunable_weights(),
            self.s3_fusion.get_prunable_weights(),
            self.s4_b1_blocks.get_prunable_weights(),
            self.s4_b2_blocks.get_prunable_weights(),
            self.s4_b3_blocks.get_prunable_weights(),
            self.s4_b4_blocks.get_prunable_weights()
        ]))

        return prunable_weights
