from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones
from .scse import scse_block

import gezi
from gezi import logging

backend = None
layers = None
models = None
keras_utils = None

# --- attention unet --------------------
# https://github.com/robinvvinod/unet/blob/master/layers2D.py
# https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
from tensorflow.keras.layers import Conv2D, Lambda, Dense, Multiply, Add, Activation, UpSampling2D, BatchNormalization
from tensorflow.keras import backend as K

def expend_as(tensor, rep):
  
    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape, kernel_size=3, strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]), padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = Add()([phi_g, theta_x])
    add_xg = Activation(gezi.get('activation') or 'relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[3])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = Multiply()([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=shape_x[3], kernel_size=1, strides=1, padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation=gezi.get('activation') or 'relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)
    add_name = 'decoder_stage{}_add'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):
        ks = gezi.get('decoder_kernel_size', 4)
        x = layers.Conv2DTranspose(
            filters,
            # kernel_size=(4, 4),  # 这了4是否是最佳 如果设置3呢。。。
            kernel_size=(ks, ks),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation(gezi.get('activation') or 'relu', name=relu_name)(x)

        skip_combiner = gezi.get('unet_skip_combiner', 'concat')
        if skip is not None:
            if skip_combiner == 'concat':
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
            else:
                skip =  Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(skip)
                x = layers.Add(name=add_name)([x, skip]) 

        if skip_combiner == 'concat':
          x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation=None,
        use_batchnorm=True,
        kernel_size=3,
        use_scse=False,
        use_attention=False,
        dropout=0.,
):
    input_ = backbone.input
    x = backbone.output
    # 这里dropout影响还挺大 是否考虑每层都加dropout ? TODO
    x = layers.Dropout(dropout)(x)
    # backbone_out = x
    
    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # 4
    # (None, 16, 16, 960)
    # (None, 32, 32, 336)
    # (None, 64, 64, 192)
    # (None, 128, 128, 144)
    # print(len(skips))
    # for x in skips:
    #     print(x.shape)
    # (None, 8, 8, 1792)
    # print(backbone_out.shape)

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

# 0 Tensor("dropout/cond/Identity:0", shape=(None, 8, 8, 512), dtype=float32) 128 Tensor("stage4_unit1_relu1/Relu:0", shape=(None, 16, 16, 256), dtype=float32)
# Tensor("decoder_stage0b_swish/mul:0", shape=(None, 16, 16, 128), dtype=float32)
# 1 Tensor("decoder_stage0b_swish/mul:0", shape=(None, 16, 16, 128), dtype=float32) 64 Tensor("stage3_unit1_relu1/Relu:0", shape=(None, 32, 32, 128), dtype=float32)
# Tensor("decoder_stage1b_swish/mul:0", shape=(None, 32, 32, 64), dtype=float32)
# 2 Tensor("decoder_stage1b_swish/mul:0", shape=(None, 32, 32, 64), dtype=float32) 32 Tensor("stage2_unit1_relu1/Relu:0", shape=(None, 64, 64, 64), dtype=float32)
# Tensor("decoder_stage2b_swish/mul:0", shape=(None, 64, 64, 32), dtype=float32)
# 3 Tensor("decoder_stage2b_swish/mul:0", shape=(None, 64, 64, 32), dtype=float32) 16 Tensor("relu0/Relu:0", shape=(None, 128, 128, 64), dtype=float32)
# Tensor("decoder_stage3b_swish/mul:0", shape=(None, 128, 128, 16), dtype=float32)
# 4 Tensor("decoder_stage3b_swish/mul:0", shape=(None, 128, 128, 16), dtype=float32) 8 None
# Tensor("decoder_stage4b_swish/mul:0", shape=(None, 256, 256, 8), dtype=float32)

# 0 KerasTensor(type_spec=TensorSpec(shape=(None, 8, 8, 1280), dtype=tf.float32, name=None), name='dropout/Identity:0', description="created by layer 'dropout'") 256 KerasTensor(type_spec=TensorSpec(shape=(None, 16, 16, 672), dtype=tf.float32, name=None), name='block6a_expand_activation/mul:0', description="created by layer 'block6a_expand_activation'")
# 1 KerasTensor(type_spec=TensorSpec(shape=(None, 16, 16, 256), dtype=tf.float32, name=None), name='decoder_stage0b_swish/mul:0', description="created by layer 'decoder_stage0b_swish'") 128 KerasTensor(type_spec=TensorSpec(shape=(None, 32, 32, 240), dtype=tf.float32, name=None), name='block4a_expand_activation/mul:0', description="created by layer 'block4a_expand_activation'")
# 2 KerasTensor(type_spec=TensorSpec(shape=(None, 32, 32, 128), dtype=tf.float32, name=None), name='decoder_stage1b_swish/mul:0', description="created by layer 'decoder_stage1b_swish'") 64 KerasTensor(type_spec=TensorSpec(shape=(None, 64, 64, 144), dtype=tf.float32, name=None), name='block3a_expand_activation/mul:0', description="created by layer 'block3a_expand_activation'")
# 3 KerasTensor(type_spec=TensorSpec(shape=(None, 64, 64, 64), dtype=tf.float32, name=None), name='decoder_stage2b_swish/mul:0', description="created by layer 'decoder_stage2b_swish'") 32 KerasTensor(type_spec=TensorSpec(shape=(None, 128, 128, 96), dtype=tf.float32, name=None), name='block2a_expand_activation/mul:0', description="created by layer 'block2a_expand_activation'")
# 4 KerasTensor(type_spec=TensorSpec(shape=(None, 128, 128, 32), dtype=tf.float32, name=None), name='decoder_stage3b_swish/mul:0', description="created by layer 'decoder_stage3b_swish'") 16 None

    # building decoder blocks
    start = len(decoder_filters) - n_upsample_blocks

    if start:
        x = skips[start - 1]
        # backbone_out = x

    if gezi.get('seg_lite'):
        x = skips[0]
        # backbone_out = x
        skips = skips[1:]
        decoder_filters = decoder_filters[1:]
        n_upsample_blocks = len(decoder_filters)

    for i in range(start, n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        
        if use_scse:
            x = scse_block(x, prefix=f'{i}')

        ## https://github.com/robinvvinod/unet/blob/master/network.py
        # attn0 = AttnGatingBlock(c3, b0, n_filters * 16)
        # u0 = transpose_block(b0, attn0, n_filters=n_filters * 8, batchnorm=batchnorm, recurrent=2)  # 64x64x64

        # 为了性能只对skip[-1] 做attention
        # print(use_attention, use_scse, skip)
        # if use_attention and i < len(skips) - 1 and skip is not None:
        if use_attention and i > 2 and skip is not None:
            # print('1-------------skip', skip)
            skip = AttnGatingBlock(skip, x, decoder_filters[i] * 2)
            # print('2-------------skip', skip)

        logging.debug(i, x, decoder_filters[i], skip)
        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    kernel_size = (kernel_size, kernel_size)

    backbone_name = gezi.get('backbone_name')
    gezi.set('seg_notop', models.Model(input_, x, name=f'sm-unet-{backbone_name}_notop'))
    notop_out = x

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=kernel_size,  # chg modify 可以设置最后kernel_size 而不是指定(3, 3) 实验3，3 还是效果好于(1,1)
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',  # TODO 是否加入类别数据 来使得by_name可以自动载入? 避免形状不一致报错 或者可以控制这个输出name
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, [x, notop_out], name=f'sm-unet-{backbone_name}')

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        upsample_blocks=None,
        decoder_use_batchnorm=True,
        kernel_size=3,
        use_scse=False,
        use_attention=False,
        dropout=0.,
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    import gezi
    if not gezi.get('backbone'):
      gezi.set('backbone', backbone)

    gezi.set('backbone_name', backbone_name)

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=upsample_blocks or len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        kernel_size=kernel_size,
        use_scse=use_scse,
        use_attention=use_attention,
        dropout=dropout,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
