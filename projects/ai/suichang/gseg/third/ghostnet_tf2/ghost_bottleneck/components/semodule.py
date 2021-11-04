from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Lambda, Reshape, Layer, Activation


class SEModule(Layer):
    """
    A squeeze and excite module
    """
    def __init__(self, filters, ratio):
        super(SEModule, self).__init__()
        self.pooling = GlobalAveragePooling2D()
        self.reshape = Lambda(self._reshape)
        self.conv1 = Conv2D(int(filters / ratio), (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)
        self.conv2 = Conv2D(int(filters), (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)
        self.relu = Activation('relu')
        self.hard_sigmoid = Activation('hard_sigmoid')

    @staticmethod
    def _reshape(x):
        return Reshape((1, 1, int(x.shape[1])))(x)

    @staticmethod
    def _excite(x, excitation):
        """
        Multiply by an excitation factor

        :param x: A Tensorflow Tensor
        :param excitation: A float between 0 and 1
        :return:
        """
        return x * excitation

    def call(self, inputs):
        x = self.reshape(self.pooling(inputs))
        x = self.relu(self.conv1(x))
        excitation = self.hard_sigmoid(self.conv2(x))
        x = Lambda(self._excite, arguments={'excitation': excitation})(inputs)
        return x
