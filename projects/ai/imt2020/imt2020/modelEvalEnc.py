#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import tensorflow as tf
from modelDesign import *
import scipy.io as sio
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 512
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('../input/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
H_test = data
#=======================================================================================================================
#=======================================================================================================================
# Model Loading and Encoding
encoder_address = '../input/modelSubmit/encoder.h5'
_custom_objects = get_custom_objects()
encModel = tf.keras.models.load_model(encoder_address, custom_objects=_custom_objects)
encode_feature = encModel.predict(H_test)
print('---------------', encode_feature.shape)
print("Feedback bits length is ", np.shape(encode_feature)[-1])
np.save('../input/encOutput.npy', encode_feature)
print('Finished!')
#=======================================================================================================================
#=======================================================================================================================
