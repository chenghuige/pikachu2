import tensorflow as tf
from tensorflow.keras import layers, regularizers



class Segception(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Segception, self).__init__(**kwargs)
        base_model = tf.keras.applications.xception.Xception(include_top=False, weights=weights,
                                                             input_shape=input_shape, pooling='avg')
        output_1 = base_model.get_layer('block2_sepconv2_bn').output
        output_2 = base_model.get_layer('block3_sepconv2_bn').output
        output_3 = base_model.get_layer('block4_sepconv2_bn').output
        output_4 = base_model.get_layer('block13_sepconv2_bn').output
        output_5 = base_model.get_layer('block14_sepconv2_bn').output
        outputs = [output_5, output_4, output_3, output_2, output_1]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # Decoder
        self.adap_encoder_1 = EncoderAdaption(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(filters=64, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(filters=256, kernel_size=3, dilation_rate=1, blocks=5)
        self.decoder_conv_2 = FeatureGeneration(filters=128, kernel_size=3, dilation_rate=1, blocks=5)
        self.decoder_conv_3 = FeatureGeneration(filters=64, kernel_size=3, dilation_rate=1, blocks=5)
        self.decoder_conv_4 = FeatureGeneration(filters=64, kernel_size=3, dilation_rate=1, blocks=2)
        self.aspp = ASPP_2(filters=64, kernel_size=3)

        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)
        self.conv_logits_aux = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = layers.LeakyReLU(alpha=0.3)(outputs[i])

        x = self.adap_encoder_1(outputs[0], training=training)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x)  # 512
        x = self.decoder_conv_1(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)  # 256
        x = self.decoder_conv_2(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)  # 128
        x = self.decoder_conv_3(x, training=training)  # 128

        x = self.aspp(x, training=training, operation='sum')  # 128
        x_aux = self.conv_logits_aux(x)
        x_aux = upsampling(x_aux, scale=2)
        x_aux_out = upsampling(x_aux, scale=2)

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
        x = self.decoder_conv_4(tf.concat((x, x_aux), -1), training=training)  # 64
        x = self.conv_logits(x)
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x_aux_out
        else:
            return x




class Efficient(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Efficient, self).__init__(**kwargs)#1024x512
        filters = 24
        self.conv1 = Conv_BN(filters=filters, kernel_size=3, strides=2) # 512x256
        self.conv2 = DepthwiseConv_BN(filters*3, kernel_size=3, dilation_rate=1, strides=2)#256x128

        self.block1 = EfficientBlock(filters*3, 3, dilation_rate=1, blocks=2)

        self.block1_dilated1 = EfficientBlock(filters*3, 3, dilation_rate=2, blocks=7)
        self.block1_dilated2 = EfficientBlock(filters*3, 3, dilation_rate=4, blocks=5)
        self.block1_dilated3 = EfficientBlock(filters*3, 3, dilation_rate=8, blocks=3)
        self.block1_dilated4 = EfficientBlock(filters*3, 3, dilation_rate=16, blocks=1)
        self.conv6 = DepthwiseConv_BN(filters*6, kernel_size=3, dilation_rate=1, strides=1)

        self.conv4 = DepthwiseConv_BN(filters*6, kernel_size=3, dilation_rate=1, strides=2)

        self.block1_dilated55 = EfficientBlock(filters*6, 3, dilation_rate=16, blocks=1)
        self.block1_dilated5 = EfficientBlock(filters*6, 3, dilation_rate=8, blocks=3)
        self.block1_dilated6 = EfficientBlock(filters*6, 3, dilation_rate=4, blocks=5)
        self.block1_dilated7 = EfficientBlock(filters*6, 3, dilation_rate=2, blocks=7)

        self.conv5 = DepthwiseConv_BN(filters*3, kernel_size=3, dilation_rate=1, strides=1)
        self.block2 = EfficientBlock(filters*3, 3, dilation_rate=1, blocks=5)


        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)


    def call(self, inputs, training=None, mask=None, aux_loss=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        #x = self.conv3(x_, training=training)

        x_block = self.block1(x, training=training)
        x_dil = self.block1_dilated1(x_block, training=training)
        x_dil = self.block1_dilated2(x_dil, training=training)
        x_dil = self.block1_dilated3(x_dil, training=training)
        x_dil_ = self.block1_dilated4(x_dil, training=training)

        x_dil = self.conv4(x_dil_, training=training)


        x_dil = self.block1_dilated55(x_dil, training=training)
        x_dil = self.block1_dilated5(x_dil, training=training)
        x_dil = self.block1_dilated6(x_dil, training=training)
        x_dil = self.block1_dilated7(x_dil, training=training)

        x = upsampling(x_dil, scale=4) + upsampling(self.conv6(x_dil_, training=training), scale=2)
        x = self.conv5(x, training=training)
        x = self.block2(x, training=training)

        x = upsampling(x, scale=2)
        x = self.conv_logits(x)
        if aux_loss:
            return x, x
        else:
            return x




class Segception_small(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Segception_small, self).__init__(**kwargs)
        base_model = tf.keras.applications.xception.Xception(include_top=False, weights=weights,
                                                             input_shape=input_shape, pooling='avg')
        output_1 = base_model.get_layer('block2_sepconv2_bn').output
        output_2 = base_model.get_layer('block3_sepconv2_bn').output
        output_3 = base_model.get_layer('block4_sepconv2_bn').output
        output_4 = base_model.get_layer('block13_sepconv2_bn').output
        output_5 = base_model.get_layer('block14_sepconv2_bn').output
        outputs = [output_5, output_4, output_3, output_2, output_1]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # Decoder
        self.adap_encoder_1 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(filters=64, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(filters=32, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(filters=128, kernel_size=3, dilation_rate=1, blocks=3)
        self.decoder_conv_2 = FeatureGeneration(filters=64, kernel_size=3, dilation_rate=1, blocks=3)
        self.decoder_conv_3 = FeatureGeneration(filters=32, kernel_size=3, dilation_rate=1, blocks=3)
        self.decoder_conv_4 = FeatureGeneration(filters=32, kernel_size=3, dilation_rate=1, blocks=1)
        self.aspp = ASPP_2(filters=32, kernel_size=3)

        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = layers.LeakyReLU(alpha=0.3)(outputs[i])

        x = self.adap_encoder_1(outputs[0], training=training)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x)  # 512
        x = self.decoder_conv_1(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)  # 256
        x = self.decoder_conv_2(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)  # 128
        x = self.decoder_conv_3(x, training=training)  # 128

        x = self.aspp(x, training=training, operation='sum')  # 128

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
        x = self.decoder_conv_4(x, training=training)  # 64
        x = self.conv_logits(x)
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x
        else:
            return x


class Dilated_net(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Dilated_net, self).__init__(**kwargs)
        base_filter = 64
        self.conv1 = Conv_BN(filters=base_filter, kernel_size=3, strides=2)
        self.encoder_conv_1 = FeatureGeneration(filters=base_filter, kernel_size=3, dilation_rate=1, blocks=2)
        self.downsample_1 = DepthwiseConv_BN(filters=base_filter, kernel_size=3, strides=2)
        self.encoder_conv_2 = FeatureGeneration(filters=base_filter*2, kernel_size=3, dilation_rate=1, blocks=4)
        self.downsample_2 = DepthwiseConv_BN(filters=base_filter*2, kernel_size=3, strides=2)
        self.encoder_conv_3 = FeatureGeneration(filters=base_filter*4, kernel_size=3, dilation_rate=1, blocks=5)
        self.encoder_conv_4 = FeatureGeneration(filters=base_filter*4, kernel_size=3, dilation_rate=2, blocks=4)
        self.encoder_conv_5 = FeatureGeneration(filters=base_filter*4, kernel_size=3, dilation_rate=4, blocks=3)
        self.encoder_conv_6 = FeatureGeneration(filters=base_filter*4, kernel_size=3, dilation_rate=8, blocks=2)
        self.encoder_conv_7 = FeatureGeneration(filters=base_filter*4, kernel_size=3, dilation_rate=16, blocks=1)

        self.adap_encoder_1 = EncoderAdaption(filters=base_filter*2, kernel_size=3, dilation_rate=1)


        #DepthwiseConv_BN
        self.decoder_conv_1 = FeatureGeneration(filters=base_filter*2, kernel_size=3, dilation_rate=1, blocks=6)
        self.decoder_conv_2 = FeatureGeneration(filters=base_filter, kernel_size=3, dilation_rate=1, blocks=3)
        self.aspp = ASPP_2(filters=base_filter*2, kernel_size=3)

        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)
        self.conv_logits_aux = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        x = self.conv1(inputs, training=training)
        x = self.encoder_conv_1(x, training=training)
        x_enc = self.downsample_1(x, training=training)
        x = self.encoder_conv_2(x_enc, training=training)
        x = self.downsample_2(x, training=training)
        x1 = self.encoder_conv_3(x, training=training)
        x = x1 + self.encoder_conv_4(x1, training=training)
        x += self.encoder_conv_5(x + x1, training=training)
        x += self.encoder_conv_6(x + x1, training=training)
        x += self.encoder_conv_7(x + x1, training=training)
        x = upsampling(x + x1, scale=2)
        x = self.decoder_conv_1(x, training=training)

        x += self.adap_encoder_1(x_enc, training=training)

        x = self.aspp(x, training=training, operation='sum')  # 128
        x_aux = self.conv_logits_aux(x)
        x_aux = upsampling(x_aux, scale=2)
        x_aux_out = upsampling(x_aux, scale=2)

        x = upsampling(x, scale=2)
        x = self.decoder_conv_2(tf.concat((x, x_aux), -1), training=training)  # 64
        x = self.conv_logits(tf.concat((x, x_aux), -1))
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x_aux_out
        else:
            return x





def upsampling(inputs, scale):

    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
                                    align_corners=True)


def reshape_into(inputs, input_to_copy):
    return tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
                                             input_to_copy.get_shape()[2].value], align_corners=True)


# convolution
def conv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=False):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0003),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)

class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None, activation=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.LeakyReLU(alpha=0.3)(x)

        return x


class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = layers.LeakyReLU(alpha=0.3)(x)

        return x


class Transpose_Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(Transpose_Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = layers.LeakyReLU(alpha=0.3)(x)

        return x


class EfficientBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1,  blocks=3):
        super(EfficientBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.first_conv = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.blocks = []
        for n in xrange(blocks - 1):
            self.blocks = self.blocks + [
                DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]

    def call(self, inputs, training=None):

        x = self.first_conv(inputs, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        return x + inputs




class ShatheBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,  dilation_rate=1, bottleneck=2):
        super(ShatheBlock, self).__init__()

        self.filters = filters * bottleneck
        self.kernel_size = kernel_size

        self.conv = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv1 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv3 = Conv_BN(filters, kernel_size=1)

    def call(self, inputs, training=None):
        x = self.conv(inputs, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x + inputs


class ShatheBlock_MultiDil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, bottleneck=2):
        super(ShatheBlock_MultiDil, self).__init__()

        self.filters = filters * bottleneck
        self.filters_dil = filters / 2
        self.kernel_size = kernel_size

        self.conv = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv1 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*8)
        self.conv2 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*4)
        self.conv3 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*6)
        self.conv4 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*2)
        self.conv5 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv6 = Conv_BN(filters, kernel_size=1)

    def call(self, inputs, training=None):
        x1 = self.conv(inputs, training=training)
        x2 = self.conv1(x1, training=training)
        x3 = self.conv2(x1, training=training)
        x4 = self.conv3(x1, training=training)
        x5 = self.conv4(x1, training=training)
        x6 = self.conv5(tf.concat((x2,x3,x4,x5), -1) + x1, training=training)
        x7 = self.conv6(x6, training=training)
        return x7 + inputs


class ASPP(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ASPP, self).__init__()

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=1, dilation_rate=1)
        self.conv2 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=4)
        self.conv3 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=8)
        self.conv4 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=16)
        self.conv5 = Conv_BN(filters, kernel_size=1)

    def call(self, inputs, training=None, operation='concat'):
        feature_map_size = tf.shape(inputs)
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
        image_features = self.conv1(image_features, training=training)
        image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
        x1 = self.conv2(inputs, training=training)
        x2 = self.conv3(inputs, training=training)
        x3 = self.conv4(inputs, training=training)
        if 'concat' in operation:
            x = self.conv5(tf.concat((image_features, x1, x2, x3, inputs), axis=3), training=training)
        else:
            x = image_features + x1 + x2 + x3 + inputs

        return x


class ASPP_2(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ASPP_2, self).__init__()

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=1, dilation_rate=1)
        self.conv2 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=4)
        self.conv3 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=8)
        self.conv4 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=16)
        self.conv6 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(2, 8))
        self.conv7 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(6, 3))
        self.conv8 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(8, 2))
        self.conv9 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(3, 6))
        self.conv5 = Conv_BN(filters, kernel_size=1)

    def call(self, inputs, training=None, operation='concat'):
        feature_map_size = tf.shape(inputs)
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
        image_features = self.conv1(image_features, training=training)
        image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
        x1 = self.conv2(inputs, training=training)
        x2 = self.conv3(inputs, training=training)
        x3 = self.conv4(inputs, training=training)
        x4 = self.conv6(inputs, training=training)
        x5 = self.conv7(inputs, training=training)
        x4 = self.conv8(inputs, training=training) + x4
        x5 = self.conv9(inputs, training=training) + x5
        if 'concat' in operation:
            x = self.conv5(tf.concat((image_features, x1, x2, x3,x4,x5, inputs), axis=3), training=training)
        else:
            x = self.conv5(image_features + x1 + x2 + x3+x5+x4, training=training) + inputs

        return x




class DPC(tf.keras.Model):
    def __init__(self, filters):
        super(DPC, self).__init__()

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(1, 6))
        self.conv2 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(18, 15))
        self.conv3 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(6, 21))
        self.conv4 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(1, 1))
        self.conv5 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(6, 3))

    def call(self, inputs, training=None, operation='concat'):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(x1, training=training)
        x3 = self.conv3(x1, training=training)
        x4 = self.conv4(x1, training=training)
        x5 = self.conv5(x2, training=training)

        if 'concat' in operation:
            x = tf.concat((x1, x2, x3, x4, x5, inputs), axis=3)
        else:
            x = x1 + x2 + x3 + x4 + x5 + inputs
        return x


class EncoderAdaption(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super(EncoderAdaption, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = Conv_BN(filters, kernel_size=1)
        self.conv2 = ShatheBlock(filters, kernel_size=kernel_size, dilation_rate=dilation_rate)

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x


class FeatureGeneration(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1,  blocks=3):
        super(FeatureGeneration, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv0 = Conv_BN(self.filters, kernel_size=1)
        self.blocks = []
        for n in xrange(blocks):
            self.blocks = self.blocks + [
                ShatheBlock(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]

    def call(self, inputs, training=None):

        x = self.conv0(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        return x






class FeatureGeneration_Dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1,  blocks=3):
        super(FeatureGeneration_Dil, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv0 = Conv_BN(self.filters, kernel_size=1)
        self.blocks = []
        for n in xrange(blocks):
            self.blocks = self.blocks + [
                ShatheBlock_MultiDil(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]

    def call(self, inputs, training=None):

        x = self.conv0(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        return x


