"""
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from .resnet_101 import resnet101_model
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization
import melt as mt

kern_init = keras.initializers.he_normal()
kern_reg = keras.regularizers.l2(1e-5)

def ResidualConvUnit(inputs,n_filters=128,kernel_size=3,name=''):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """
    
    net = ReLU(name=name+'relu1')(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = ReLU(name=name+'relu2')(net)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = Add(name=name+'sum')([net, inputs])
    
    return net

def ChainedResidualPooling(inputs,n_filters=256,name=''):
    """
    Chained residual pooling aims to capture background 
    context from a large image region. This component is 
    built as a chain of 2 pooling blocks, each consisting 
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are 
    fused together with the input feature map through summation 
    of residual connections.

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv

    Returns:
      Double-pooled feature maps
    """
    
    net = ReLU(name=name+'relu')(inputs)
    net_out_1 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool1', data_format='channels_last')(net)
    net_out_2 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool2', data_format='channels_last')(net)
    net_out_3 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool3', data_format='channels_last')(net)
    net_out_4 = net
    
    net = Conv2D(n_filters, 3, padding='same', name=name+'conv4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool4', data_format='channels_last')(net)
    net_out_5 = net
    
    net = Add(name=name+'sum')([net_out_1,net_out_2,net_out_3,net_out_4,net_out_5])

    return net


def MultiResolutionFusion(high_inputs=None,low_inputs=None,n_filters=128,name=''):
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension 
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.

    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv

    Returns:
      Fused feature maps at higher resolution
    
    """
    
    if low_inputs is None: # RefineNet block 4
        return high_inputs

    else:
        conv_low = Conv2D(n_filters, 3, padding='same', name=name+'conv_lo', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(low_inputs)
        conv_low = BatchNormalization()(conv_low)
        conv_high = Conv2D(n_filters, 3, padding='same', name=name+'conv_hi', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high_inputs)
        conv_high = BatchNormalization()(conv_high)
        
        conv_low_up = UpSampling2D(size=2, interpolation='bilinear', name=name+'up')(conv_low)
        
        return Add(name=name+'sum')([conv_low_up, conv_high])


def RefineBlock(high_inputs=None,low_inputs=None,block=0):
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.

    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution

    Returns:
      RefineNet block for a single path i.e one resolution
    
    """

    if low_inputs is None: # block 4
        rcu_high = ResidualConvUnit(high_inputs, n_filters=256, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters=256, name='rb_{}_rcu_h2_'.format(block))
        
        # nothing happens here
        fuse = MultiResolutionFusion(high_inputs = rcu_high,
                                     low_inputs = None,
                                     n_filters=256,
                                     name = 'rb_{}_mrf_'.format(block))
        
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256, name='rb_{}_crp_'.format(block))
        
        output = ResidualConvUnit(fuse, n_filters=256, name='rb_{}_rcu_o1_'.format(block))
        return output
    else:
        high_n = K.int_shape(high_inputs)[-1]
        low_n = K.int_shape(low_inputs)[-1]
        
        rcu_high = ResidualConvUnit(high_inputs, n_filters = high_n, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters = high_n, name='rb_{}_rcu_h2_'.format(block))
        
        rcu_low = ResidualConvUnit(low_inputs, n_filters = low_n, name='rb_{}_rcu_l1_'.format(block))
        rcu_low = ResidualConvUnit(rcu_low, n_filters = low_n, name='rb_{}_rcu_l2_'.format(block))

        fuse = MultiResolutionFusion(high_inputs = rcu_high,
                                     low_inputs = rcu_low,
                                     n_filters=128,
                                     name = 'rb_{}_mrf_'.format(block))
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=128, name='rb_{}_crp_'.format(block))
        output = ResidualConvUnit(fuse_pooling, n_filters=128, name='rb_{}_rcu_o1_'.format(block))
        return output

def _get_layernames(backbone):
  # (16,16) (64,64)
  backbone=backbone.lower()
  if backbone.startswith('eff'):
    return ['block6a_expand_activation', 'block4a_expand_activation', 'block3a_expand_activation', 'block2a_expand_activation']
  else:
    raise ValueError(backbone)

def build_refinenet(input_shape, num_class, backbone='EfficientNetB4', weights='noisy-student', frontend_trainable=True):
    """
    Builds the RefineNet model. 

    Arguments:
      input_shape: Size of input image, including number of channels
      num_classes: Number of classes
      resnet_weights: Path to pre-trained weights for ResNet-101
      frontend_trainable: Whether or not to freeze ResNet layers during training

    Returns:
      RefineNet model
    """
    # Build ResNet-101
    # model_base = resnet101_model(input_shape, None)
    # # Get ResNet block output layers
    # high = model_base.output
    # print(high)
    # exit(0)

    inputs = tf.keras.Input(input_shape[-3:])
    Model_ = mt.image.get_classifier(backbone)
    model = Model_(input_tensor=inputs, weights=weights, include_top=False)
    backbone_out = model.output

    # Get ResNet block output layers
    layer_names = _get_layernames(backbone)
    high = [model.get_layer(layer_name).output for layer_name in layer_names]
    low = [None, None, None]

    # print('--------high', high)

    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(256, 1, padding='same', name='resnet_map1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[0])
    high[1] = Conv2D(128, 1, padding='same', name='resnet_map2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[1])
    high[2] = Conv2D(128, 1, padding='same', name='resnet_map3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[2])
    high[3] = Conv2D(128, 1, padding='same', name='resnet_map4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[3])
    print('high', high)
    for h in high:
        h = BatchNormalization()(h)

    # RefineNet
    low[0] = RefineBlock(high_inputs = high[0], low_inputs = None, block=4) # Only input ResNet 1/32
    low[1] = RefineBlock(high_inputs = high[1], low_inputs = low[0], block=3) # High input = ResNet 1/16, Low input = Previous 1/16
    low[2] = RefineBlock(high_inputs = high[2], low_inputs = low[1], block=2) # High input = ResNet 1/8, Low input = Previous 1/8
    print('low', low)
    net = RefineBlock(high_inputs = high[3], low_inputs = low[2], block=1) # High input = ResNet 1/4, Low input = Previous 1/4.

    net = ResidualConvUnit(net, name='rf_rcu_o1_')
    net = ResidualConvUnit(net, name='rf_rcu_o2_')
    
    net = UpSampling2D(size=2, interpolation='bilinear', name='rf_up_o')(net)
    notop_out = net
    net = Conv2D(num_class, 1, activation=None, name='rf_pred')(net)
    
    model = Model(inputs, [net, backbone_out, notop_out], name=f'Refinenet_{backbone}')
    
    for layer in model.layers:
        if 'rb' in layer.name or 'rf_' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = frontend_trainable

    return model