from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Layer, BatchNormalization, Activation, add

from ghost_bottleneck.components.semodule import SEModule
from ghost_bottleneck.components.ghostmodule import GhostModule


class GBNeck(Layer):
    """
    The GhostNet Bottleneck
    """
    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(GBNeck, self).__init__()
        self.strides = strides
        self.use_se = use_se
        self.conv = Conv2D(out, (1, 1), strides=(1, 1), padding='same',
                           activation=None, use_bias=False)
        self.relu = Activation('relu')
        self.depthconv1 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         activation=None, use_bias=False)
        self.depthconv2 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         activation=None, use_bias=False)
        for i in range(5):
            setattr(self, f"batchnorm{i+1}", BatchNormalization())
        self.ghost1 = GhostModule(exp, ratio, 1, 3)
        self.ghost2 = GhostModule(out, ratio, 1, 3)
        self.se = SEModule(exp, ratio)

    def call(self, inputs):
        x = self.batchnorm1(self.depthconv1(inputs))
        x = self.batchnorm2(self.conv(x))

        y = self.relu(self.batchnorm3(self.ghost1(inputs)))
        # Extra depth conv if strides higher than 1
        if self.strides > 1:
            y = self.relu(self.batchnorm4(self.depthconv2(y)))
        # Squeeze and excite
        if self.use_se:
            y = self.se(y)
        y = self.batchnorm5(self.ghost2(y))
        # Skip connection
        return add([x, y])
