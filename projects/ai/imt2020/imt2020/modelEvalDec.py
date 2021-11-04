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
mat = sio.loadmat('channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
H_test = data
# encOutput Loading
encode_feature = np.load('./encOutput.npy')
#=======================================================================================================================
#=======================================================================================================================
# Model Loading and Decoding
decoder_address = './modelSubmit/decoder.h5'
_custom_objects = get_custom_objects()
model_decoder = tf.keras.models.load_model(decoder_address, custom_objects=_custom_objects)
H_pre = model_decoder.predict(encode_feature)
if (NMSE(H_test, H_pre) < 0.1):
    print('Valid Submission')
    print('The Score is ' + np.str(1.0 - NMSE(H_test, H_pre)))
print('Finished!')
#=======================================================================================================================
#=======================================================================================================================
