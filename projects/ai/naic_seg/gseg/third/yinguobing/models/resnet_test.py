import unittest

import tensorflow as tf

from resnet import (ResNet, bottleneck_block, bottleneck_blocks, make_resnet,
                    residual_block, residual_blocks, rsn_stem, rsn_head)

devices = tf.config.get_visible_devices("CPU")
tf.config.set_visible_devices(devices)


class TestTensorShapes(unittest.TestCase):

    def setUp(self):
        self.height = 256
        self.width = 256
        self.filters = 32
        self.inputs = tf.keras.Input((self.height, self.width, self.filters))

    def test_residual_block(self):
        outputs = residual_block(filters=self.filters,
                                 downsample=False,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters])

    def test_residual_block_downsample(self):
        outputs = residual_block(filters=self.filters,
                                 kernel_size=(3, 3),
                                 downsample=True,
                                 padding='same',
                                 activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/2, self.width/2, self.filters])

    def test_residual_blocks(self):
        outputs = residual_blocks(num_blocks=2,
                                  filters=self.filters,
                                  downsample=False,
                                  activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters])

    def test_residual_blocks_downsample(self):
        outputs = residual_blocks(num_blocks=2,
                                  filters=self.filters,
                                  downsample=True,
                                  activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/2, self.width/2, self.filters])

    def test_bottleneck_block(self):
        outputs = bottleneck_block(filters=self.filters,
                                   kernel_size=(3, 3),
                                   downsample=False,
                                   padding='same',
                                   activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters*4])

    def test_bottleneck_blocks(self):
        outputs = bottleneck_blocks(num_blocks=3,
                                    filters=self.filters,
                                    downsample=False,
                                    activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height, self.width, self.filters*4])

    def test_bottleneck_block_downsample(self):
        outputs = bottleneck_block(filters=self.filters,
                                   kernel_size=(3, 3),
                                   downsample=True,
                                   padding='same',
                                   activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/2, self.width/2, self.filters*4])

    def test_bottleneck_blocks_downsample(self):
        outputs = bottleneck_blocks(num_blocks=3,
                                    filters=self.filters,
                                    downsample=True,
                                    activation='relu')(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/2, self.width/2, self.filters*4])

    def test_rsn_stem(self):
        outputs = rsn_stem(filters=self.filters,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           pool_size=(3, 3))(self.inputs)
        self.assertListEqual(outputs.shape.as_list(),
                             [None, self.height/4, self.width/4, self.filters])

    def test_rsn_head(self):
        outputs = rsn_head(1024)(self.inputs)
        self.assertListEqual(outputs.shape.as_list(), [None, 1024])


if __name__ == '__main__':
    unittest.main()

    # RESNET 18
    res18_layer_config = [2, 2, 2, 2]
    input_shape = (256, 256, 3)

    # Functional API.
    resnet18_func = make_resnet(res18_layer_config,
                                bottleneck=False,
                                input_shape=input_shape,
                                output_size=1000,
                                name="resnet_18_func")
    resnet18_func.summary()
    resnet18_func.save("./saved_model/res18_func")

    # Sub classed.
    resnet18_subclassed = ResNet(res18_layer_config,
                                 bottleneck=False,
                                 output_size=1000,
                                 name="resnet_18_subc")
    resnet18_subclassed.prepare_summary(input_shape)
    resnet18_subclassed.summary()
    resnet18_subclassed.predict(tf.zeros((1, 224, 224, 3)))
    resnet18_subclassed.save("./saved_model/res18_subc")

    # ResNet-50
    res50_layer_config = [3, 4, 6, 3]

    # Functional API
    resnet50_func = make_resnet(res50_layer_config,
                                bottleneck=True,
                                input_shape=input_shape,
                                output_size=1000,
                                name="resnet_50_func")
    resnet50_func.summary()
    resnet50_func.save("./saved_model/res50_func")

    # Sub classed
    resnet50_subclassed = ResNet(res50_layer_config,
                                 bottleneck=True,
                                 output_size=1000,
                                 name="resnet_50_subc")

    resnet50_subclassed.prepare_summary(input_shape)
    resnet50_subclassed.summary()
    resnet50_subclassed.predict(tf.zeros((1, 224, 224, 3)))
    resnet50_subclassed.save("./saved_model/res50_subc")
