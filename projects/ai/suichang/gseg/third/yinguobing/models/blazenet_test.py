import unittest

import tensorflow as tf

from blazenet import blaze_block, zero_channel_padding, double_blaze_block


class TestTensorShapes(unittest.TestCase):

    def setUp(self):
        self.height = 256
        self.width = 256
        self.filters = 32
        self.inputs = tf.keras.Input((self.height, self.width, self.filters))

    def test_channel_padding(self):
        channels_to_pad = 42
        outputs = zero_channel_padding(channels_to_pad)(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters + channels_to_pad])

    def test_blaze_block_downsample(self):
        outputs = blaze_block(self.filters + 3, strides=(2, 2))(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/2, self.width/2, self.filters + 3])

    def test_blaze_block_padding_up(self):
        outputs = blaze_block(self.filters + 3)(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters + 3])

    def test_blaze_block_padding_down(self):
        outputs = blaze_block(self.filters - 3)(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters - 3])

    def test_double_blaze_block_downsample(self):
        outputs = double_blaze_block(
            self.filters - 3, self.filters + 3, (2, 2))(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/2, self.width/2, self.filters + 3])


if __name__ == "__main__":
    unittest.main()
