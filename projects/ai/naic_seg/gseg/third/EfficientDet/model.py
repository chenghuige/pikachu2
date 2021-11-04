from functools import reduce

# from keras import layers
# from keras import initializers
# from keras import models
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend as K

# import efficientnet.tfkeras as eff
# from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from efficientnet.tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from .tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from .tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from .layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
from .initializers import PriorProbability
from .utils.anchors import anchors_for_shape
from tensorflow.keras import layers
import numpy as np

import gezi
from gezi import logging
import melt as mt
from melt.image import resize

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
# w_bifpns = [x // 2 for x in w_bifpns]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]

MOMENTUM = 0.997
EPSILON = 1e-4


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=f'{name}/conv')
    f2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=True, name='{}_conv'.format(name))
    f2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='{}_bn'.format(name))
    # f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        # P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = resize(P7_in, P6_in.shape[1:3])
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = layers.Activation('swish')(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = resize(P6_td, P5_in_1.shape[1:3])
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = layers.Activation('swish')(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = resize(P5_td, P4_in_1.shape[1:3])
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = layers.Activation('swish')(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = resize(P4_td, P3_in.shape[1:3])
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = layers.Activation('swish')(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = layers.Activation('swish')(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = layers.Activation('swish')(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation('swish')(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = layers.Activation('swish')(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = resize(P7_in, P6_in.shape[1:3])
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = layers.Activation('swish')(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P6_U = resize(P6_td, P5_in.shape[1:3])
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
        P5_td = layers.Activation('swish')(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = resize(P5_td, P4_in.shape[1:3])
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
        P4_td = layers.Activation('swish')(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = resize(P4_td, P3_in.shape[1:3])
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = layers.Activation('swish')(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = layers.Activation('swish')(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = layers.Activation('swish')(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation('swish')(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = layers.Activation('swish')(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
    return P3_out, P4_td, P5_td, P6_td, P7_out

def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = resize(P7_in, P6_in.shape[1:3])
        P6_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = layers.Activation('swish')(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = resize(P6_td, P5_in_1.shape[1:3])
        P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = layers.Activation('swish')(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = resize(P5_td, P5_U.shape[1:3])
        P4_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = layers.Activation('swish')(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = resize(P4_td, P3_in.shape[1:3])
        P3_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = layers.Activation('swish')(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = layers.Activation('swish')(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = layers.Activation('swish')(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation('swish')(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = layers.Activation('swish')(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = resize(P7_in, P6_in.shape[1:3])
        P6_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = layers.Activation('swish')(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P6_U = resize(P6_td, P5_in.shape[1:3])
        P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
        P5_td = layers.Activation('swish')(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = resize(P5_td, P4_in_1.shape[1:3])
        P4_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
        P4_td = layers.Activation('swish')(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = resize(P4_td, P3_in.shape[1:3])
        P3_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = layers.Activation('swish')(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = layers.Activation('swish')(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = layers.Activation('swish')(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation('swish')(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = layers.Activation('swish')(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
    return P3_out, P4_td, P5_td, P6_td, P7_out


class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, separable_conv=True, freeze_bn=False, detect_quadrangle=False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        self.detect_quadrangle = detect_quadrangle
        num_values = 9 if detect_quadrangle else 4
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in
                          range(depth)]
            self.head = layers.SeparableConv2D(filters=num_anchors * num_values,
                                               name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.Conv2D(filters=num_anchors * num_values, name=f'{self.name}/box-predict', **options)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/box-{i}-bn-{j}') for j in
             range(3, 8)]
            for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = layers.Lambda('swish')
        self.reshape = layers.Reshape((-1, num_values))
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs


class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, freeze_bn=False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                 **options)
                          for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                        **options)
                          for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}') for j
             in range(3, 8)]
            for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = layers.Lambda('swish')
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        self.level += 1
        return outputs

## TODO depend on automl 当前有一些import 问题 暂时重复放到这里
efficientdet_model_param_dict = {
    'efficientdet-d0':
        dict(
            name='efficientdet-d0',
            backbone_name='efficientnet-b0',
            image_size=512,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    'efficientdet-d1':
        dict(
            name='efficientdet-d1',
            backbone_name='efficientnet-b1',
            image_size=640,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    'efficientdet-d2':
        dict(
            name='efficientdet-d2',
            backbone_name='efficientnet-b2',
            image_size=768,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
        ),
    'efficientdet-d3':
        dict(
            name='efficientdet-d3',
            backbone_name='efficientnet-b3',
            image_size=896,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    'efficientdet-d4':
        dict(
            name='efficientdet-d4',
            backbone_name='efficientnet-b4',
            image_size=1024,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d5':
        dict(
            name='efficientdet-d5',
            backbone_name='efficientnet-b5',
            image_size=1280,
            fpn_num_filters=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d6':
        dict(
            name='efficientdet-d6',
            backbone_name='efficientnet-b6',
            image_size=1280,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
    'efficientdet-d7':
        dict(
            name='efficientdet-d7',
            backbone_name='efficientnet-b6',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
    'efficientdet-d7x':
        dict(
            name='efficientdet-d7x',
            backbone_name='efficientnet-b7',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=4.0,
            max_level=8,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
}

@tf.keras.utils.register_keras_serializable()
class SegmentationHead(tf.keras.layers.Layer):
  """Keras layer for semantic segmentation head."""

  def __init__(self,
               num_classes,
               num_filters,
               min_level,
               max_level,
               data_format,
               is_training_bn,
               act_type,
               strategy,
               head_strategy='transpose',
               upsampling_last=False,
               **kwargs):
    """Initialize SegmentationHead.

    Args:
      num_classes: number of classes.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      data_format: string of 'channel_first' or 'channels_last'.
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      **kwargs: other parameters.
    """
    super(SegmentationHead, self).__init__(**kwargs)
    from  gseg.third.automl.efficientdet.keras import util_keras

    self.num_classes = num_classes
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.data_format = data_format
    self.is_training_bn = is_training_bn
    self.act_type = act_type
    self.strategy = strategy
    self.head_strategy = head_strategy
    self.upsampling_last = upsampling_last

    self.act_type = act_type
    self.con2d_ts = []
    self.con2d_t_bns = []
    # print('------------------------', min_level, max_level)
    ## 注意这里模仿unet多一次transpose 但是事实上和直接原有最后image resize * 2效果几乎一样 速度也差不多
    if not gezi.get('COMPAT_OLD'):
      max_level += 1  # chg 20201205 add just like unet
    
    ks = gezi.get('decoder_kernel_size', 3)
    head_stride = gezi.get('head_stride', 2) # might be 4 (4 bad result)

    for i in range(max_level - min_level):
      self.con2d_ts.append(
          tf.keras.layers.Conv2DTranspose(
              num_filters,
              ks,
              strides=2,
              padding='same',
              data_format=data_format,
              use_bias=False,
              name=f'cond2d_ts_{i}'))
      self.con2d_t_bns.append(
         util_keras.build_batch_norm(
              is_training_bn=is_training_bn,
              data_format=data_format,
              strategy=strategy,
              name=f'bn_{i}'))

    # TODO 封装一个upsampling 传递参数 upsampling,resize,dconv/transpose
    self.head_process = None
    head_strategy = self.head_strategy
    if head_strategy == 'transpose':
        self.head_process  = tf.keras.layers.Conv2DTranspose(
            num_classes, ks, strides=head_stride, padding='same', name='head_transpose')
        self.head_conv = None
    elif head_strategy == 'upsampling':
        scale_size = 4
        self.head_process = tf.keras.layers.UpSampling2D(size=scale_size, interpolation='bilinear')
        self.head_conv = tf.keras.layers.Conv2D(num_classes, kernel_size=(1,1), padding='same')
    elif head_strategy == 'resize':
        image_size = gezi.get('image_size')
        assert image_size
        self.head_process = tf.keras.layers.Lambda(lambda x: resize(x, image_size))
        self.head_conv = tf.keras.layers.Conv2D(num_classes, kernel_size=(1,1), padding='same')
    else:
        raise ValueError(head_strategy)

    dropout = gezi.get('effdet_dropout', flags=True)
    if dropout:
        self.dropout = tf.keras.layers.Dropout(float(dropout))
    else:
        self.dropout = None

    logging.debug('head_strategy:', head_strategy, self.head_process, 'upsampling_last:', upsampling_last, 'dropout:', self.dropout, dropout)


  # NotImplementedError: Layer SegmentationHead has arguments in `__init__` and therefore must override `get_config`. 如果没有get_config
  def get_config(self):
    config = {
        'num_classes': self.num_classes,
        'num_filters': self.num_filters,
        'min_level': self.min_level,
        'max_level': self.max_level,
        'data_format': self.data_format,
        'is_training_bn': self.is_training_bn,
        'act_type': self.act_type,
        'strategy': self.strategy,
        'head_strategy': self.head_strategy,
        'upsampling_last': self.upsampling_last,
        }
    base_config = super(SegmentationHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, feats, training=False):
    # print('---------', len(feats))
    # print([(x.shape, x.name) for x in feats])
    # print(feats)    
    x = feats[0] # TODO dropout here like unet ?

    if self.dropout is not None:
        x = self.dropout(x)

    skips = feats[1:]
    # print([(x.shape, x.name) for x in skips])

    # print(len(skips), len(self.con2d_t_bns))
    if len(skips) < len(self.con2d_t_bns):
      skips.append(None)
    if len(skips) < len(self.con2d_t_bns):
        self.con2d_t_bns = self.con2d_t_bns[:len(skips)]
    # print(len(skips), len(self.con2d_t_bns))
    # print(skips)
    # print(self.con2d_t_bns)

    for con2d_t, con2d_t_bn, skip in zip(self.con2d_ts, self.con2d_t_bns,
                                         skips):
    #   print(x.shape, skip.shape if skip is not None else None, con2d_t)
      if skip is not None and x.shape[1] * 2 != skip.shape[1]:
        x = resize(x, skip.shape[1:3])
      else:
        x = con2d_t(x)
      x = con2d_t_bn(x, training)
      from  gseg.third.automl.efficientdet import utils
      ## TODO下面写法save graph有点问题 其实最终实际等价标准tf.nn.swish
      #   x = utils.activation_fn(x, self.act_type)
      x = tf.keras.layers.Activation(self.act_type)(x)
      if skip is not None:
        x = tf.concat([x, skip], axis=-1)
    #   print('------output', x.shape)

    # This is the last layer of the model
    # 如果是transpose注意 最后到128 后面还需要resize一次，如果是upsampling或者resize 这里直接resize到输出大小 *4

    notop = x
    if self.head_conv is not None and not self.upsampling_last:
      x = self.head_conv(x)

    x = self.head_process(x)  # 64x64 -> 128x128

    if self.head_conv is not None and self.upsampling_last:
      x = self.head_conv(x)

    # print('------------final shape', x.shape)
    return x, notop

def efficientdet(phi, input_shape, num_classes=20, num_anchors=9, weighted_bifpn=True, freeze_bn=False,
                 score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True, 
                 weights='noisy-student', head_strategy='transpose', upsampling_last=False, 
                 start_level=0, bifpn=None, custom_backbone=True):

    assert phi in range(7)
    # input_size = image_sizes[phi]
    # input_shape = (input_size, input_size, 3)
    # image_input2 = None
    # if not gezi.get('effdet_feats'):
    image_input = layers.Input(input_shape) 
    # else:
    #     image_input2 = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = d_bifpns[phi] if bifpn is None else bifpn
    w_head = w_bifpn
    d_head = d_heads[phi]

    ## 注意这里有一个选择如果用标准颁布的efficientnet库的effnet backbone实验效果会差一点 大概稳定降低 0.002
    ## 但是模型大小会小一点 综合反而可能有收益 需要判断 按照模型大小/2之后计算模型变小的收益加上速度也变快的收益大于指标微弱降低的损失
    ## 追求模型效果的时候选用自带eff 反之选用标准 但是注意很奇怪的是b0的大小是一致的 b3 才会标准eff小一些 所以b0 用custom比较好
    if custom_backbone:
      backbone_cls = backbones[phi]
      logging.info('using custom backbone', backbone_cls)
    else:
      backbone_cls = mt.get_classifier(f'EfficientNetB{phi}')
      logging.info('using standard backbone', backbone_cls)
    
    # 本地efficientnet 有freeze_bn参数 直接返回features,修改支持nosisy student权重
    # TODO 这个微调的本地eff 载入noisy student权重是否有问题 效果？ 考虑还是用标准eff 获取相应layer ？ 似乎效果ok
    # features = gezi.get('effdet_feats')
    # if features is None:
    if custom_backbone:
      features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn, weights=weights).output
    else:
      backbone = backbone_cls(input_tensor=image_input, weights=weights)
      feature_names = ['block1a_project_bn', 'block2b_add', 'block3b_add', 'block5c_add', 'block7a_project_bn']
      features = [backbone.get_layer(name=x).output for x in feature_names]

    #     gezi.set('effdet_feats', features)
    # else:
    #     features = backbone_cls(input_tensor=image_input2, freeze_bn=freeze_bn, weights=weights).output

    # backbone_out = features[-1]

    # print('--------backbone_features')
    # print(features)
    # print(len(features))

    # 安装的efficientnet 1.1.1
    # features = backbone_cls(input_tensor=image_input, weights=weights)
    if weighted_bifpn:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_wBiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_BiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)

    # print('-----------fpn features')
    # print(fpn_features)
    # print(len(fpn_features))


    # box_net = BoxNet(w_head, d_head, num_anchors=num_anchors, separable_conv=separable_conv, freeze_bn=freeze_bn,
    #                  detect_quadrangle=detect_quadrangle, name='box_net')
    # class_net = ClassNet(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors,
    #                      separable_conv=separable_conv, freeze_bn=freeze_bn, name='class_net')
    # classification = [class_net([feature, i]) for i, feature in enumerate(fpn_features)]
    # classification = layers.Concatenate(axis=1, name='classification')(classification)
    # regression = [box_net([feature, i]) for i, feature in enumerate(fpn_features)]
    # regression = layers.Concatenate(axis=1, name='regression')(regression)

    # model = models.Model(inputs=[image_input], outputs=[classification, regression], name='efficientdet')
    
    # from gseg.third.automl.efficientdet.keras.efficientdet_keras import SegmentationHead
    # from gseg.third.automl.efficientdet.hpparams_config import efficientdet_model_param_dict
    filters = efficientdet_model_param_dict[f'efficientdet-d{phi}']['fpn_num_filters']
    feats = list(reversed(fpn_features))
    feats = feats[start_level:]
    max_level = 7 - start_level
    seg_head = SegmentationHead(num_classes, filters, 3, max_level, 'channels_last', not freeze_bn, 'swish', None, 
                                head_strategy=head_strategy, upsampling_last=upsampling_last, 
                                name='seg_head')
    outputs, notop = seg_head(feats, K.learning_phase())

    # print(outputs)
    # print(outputs.shape)
    
    # exit(0)
    # if image_input2 is None:
    model = models.Model(image_input, [outputs, notop], name='efficientdet')
    # else:
    #     model = models.Model(image_input2, [outputs, backbone_out, notop])
    return model

    # apply predicted regression to anchors
    # anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    # anchors_input = np.expand_dims(anchors, axis=0)
    # boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    # boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # # filter detections (apply NMS / score threshold / select top-k)
    # if detect_quadrangle:
    #     detections = FilterDetections(
    #         name='filtered_detections',
    #         score_threshold=score_threshold,
    #         detect_quadrangle=True
    #     )([boxes, classification, regression[..., 4:8], regression[..., 8]])
    # else:
    #     detections = FilterDetections(
    #         name='filtered_detections',
    #         score_threshold=score_threshold
    #     )([boxes, classification])

    # prediction_model = models.Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    # return model, prediction_model


if __name__ == '__main__':
    x, y = efficientdet(1)
