from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, Reshape

from ghost_bottleneck.bottleneck import GBNeck

# https://github.com/iamhankai/ghostnet.pytorch/issues/35
# 我们选了 [3,5,11,16] 这些block作为FPN的输入。超参设置和你训mnasnet一样就行吧。

class GhostNet(Model):
    """
    The main GhostNet architecture as specified in "GhostNet: More Features from Cheap Operations"
    Paper:
    https://arxiv.org/pdf/1911.11907.pdf
    """
    def __init__(self, classes):
        super(GhostNet, self).__init__()
        self.classes = classes
        self.conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                            activation=None, use_bias=False)
        self.conv2 = Conv2D(960, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                            activation=None, use_bias=False)
        self.conv3 = Conv2D(1280, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.conv4 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        for i in range(3):
            setattr(self, f"batchnorm{i+1}", BatchNormalization())
        self.relu = Activation('relu')
        self.softmax = Activation('softmax')
        self.squeeze = Lambda(self._squeeze)
        self.reshape = Lambda(self._reshape)
        self.pooling = GlobalAveragePooling2D()

        self.dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
        self.exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
        self.outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        self.ratios = [2] * 16
        self.use_ses = [False, False, False, True, True, False, False, False,
                        False, True, True, True, False, True, False, True]
        for i, args in enumerate(zip(self.dwkernels, self.strides, self.exps, self.outs, self.ratios, self.use_ses)):
            setattr(self, f"gbneck{i}", GBNeck(*args))

    @staticmethod
    def _squeeze(x):
        """
        Remove all axes with a dimension of 1
        """
        return K.squeeze(x, 1)

    @staticmethod
    def _reshape(x):
        return Reshape((1, 1, int(x.shape[1])))(x)

    def call(self, inputs):
        x = self.relu(self.batchnorm1(self.conv1(inputs)))
        # Iterate through Ghost Bottlenecks
        for i in range(16):
            x = getattr(self, f"gbneck{i}")(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.reshape(self.pooling(x))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.conv4(x)
        x = self.squeeze(x)
        output = self.softmax(x)
        return output
