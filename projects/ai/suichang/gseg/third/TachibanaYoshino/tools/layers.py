
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


class GlobalAveragePooling2D(layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return backend.mean(inputs, axis=[1, 2], keepdims=True)

    def compute_output_shape(self, input_shape):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
        else:
            input_shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])

    def get_config(self):
        config = super(GlobalAveragePooling2D, self).get_config()
        config['keep_dims'] = self.keep_dims
        return config


class PixelShuffle(layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        if isinstance(block_size, int):
            self.block_size = block_size
        elif isinstance(block_size, (list, tuple)):
            assert len(block_size) == 2 and block_size[0] == block_size[1]
            self.block_size = block_size[0]
        else:
            raise ValueError('error \'block_size\'.')

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        _, h, w, c = input_shape

        new_h = h * self.block_size
        new_w = w * self.block_size
        new_c = c / self.block_size ** 2

        return tf.TensorShape([input_shape[0], new_h, new_w, new_c])

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config['block_size'] = self.block_size
        return config
