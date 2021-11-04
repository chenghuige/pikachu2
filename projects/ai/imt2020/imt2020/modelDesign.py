"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""

#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_FEEDBACK_BITS = 768
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, (8-B):]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)
# Bit to Number Function Defining
def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)
    dim = result.shape[1]
    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)
    result = tf.reshape(result, [-1, NUM_FEEDBACK_BITS])
    def custom_grad(dy):
        grad = dy
        return (grad, grad)
    return result, custom_grad
class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()
    def call(self, x):
        return QuantizationOp(x, self.B)
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config
@tf.custom_gradient
def DequantizationOp(x, B):
    dim = x.shape[1]
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    x = tf.reshape(x, (-1, 128, 1))
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)
    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)
    return result, custom_grad
class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()
    def call(self, x):
        return DequantizationOp(x, self.B)
    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config
#=======================================================================================================================
#=======================================================================================================================
# Encoder and Decoder Function Defining
def Encoder(enc_input,num_feedback_bits):
    num_quan_bits = int(NUM_FEEDBACK_BITS / 128)
    def add_common_layers(y):
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.LeakyReLU()(y)
        return y
    h = layers.Conv2D(3, (3, 3), padding='SAME', data_format='channels_last')(enc_input)
    h = add_common_layers(h)
    h = layers.Flatten()(h)
    h = layers.Dense(768, activation='sigmoid')(h)
    h = layers.Dense(units=int(num_feedback_bits / num_quan_bits), activation='sigmoid')(h)
    enc_output = QuantizationLayer(num_quan_bits)(h)
    return enc_output
def Decoder(dec_input,num_feedback_bits):
    num_quan_bits = int(NUM_FEEDBACK_BITS / 128)
    def add_common_layers(y):
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.LeakyReLU()(y)
        return y
    h = DeuantizationLayer(num_quan_bits)(dec_input)
    h = tf.keras.layers.Reshape((-1, int(num_feedback_bits/num_quan_bits)))(h)
    h = layers.Dense(768, activation='sigmoid')(h)
    h = layers.Reshape((24, 16, 2))(h)
    res_h = h
    h = layers.Conv2D(3, (3, 3), padding='SAME', data_format='channels_last')(h)
    h = keras.layers.LeakyReLU()(h)
    for i in range(1):
        x = layers.Conv2D(3, kernel_size=(3, 3), padding='same', data_format='channels_last')(h)
        x = add_common_layers(x)
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_last')(h)
    dec_output = keras.layers.Add()([res_h, h])
    return dec_output
#=======================================================================================================================
#=======================================================================================================================
# NMSE Function Defining
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}
#=======================================================================================================================
#=======================================================================================================================
