#=======================================================================================================================
#=======================================================================================================================
import numpy as np
from tensorflow import keras
from modelDesign import Encoder, Decoder, NMSE#*
import scipy.io as sio
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 768
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('./channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
#=======================================================================================================================
#=======================================================================================================================
# Model Constructing
# Encoder
encInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
encOutput = Encoder(encInput, NUM_FEEDBACK_BITS)
encModel = keras.Model(inputs=encInput, outputs=encOutput, name='Encoder')
# Decoder
decInput = keras.Input(shape=(NUM_FEEDBACK_BITS,))
decOutput = Decoder(decInput, NUM_FEEDBACK_BITS)
decModel = keras.Model(inputs=decInput, outputs=decOutput, name="Decoder")
# Autoencoder
autoencoderInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
autoencoderOutput = decModel(encModel(autoencoderInput))
autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=autoencoderOutput, name='Autoencoder')
# Comliling
autoencoderModel.compile(optimizer='adam', loss='mse')
print(autoencoderModel.summary())
#=======================================================================================================================
#=======================================================================================================================
# Model Training
autoencoderModel.fit(x=data, y=data, batch_size=64, epochs=10, verbose=1, validation_split=0.05)
#=======================================================================================================================
#=======================================================================================================================
# Model Saving
# Encoder Saving
encModel.save('./modelSubmit/encoder.h5')
# Decoder Saving
decModel.save('./modelSubmit/decoder.h5')
#=======================================================================================================================
#=======================================================================================================================
# Model Testing
H_test = data
H_pre = autoencoderModel.predict(H_test, batch_size=512)
print('NMSE = ' + np.str(NMSE(H_test, H_pre)))
print('Training finished!')
#=======================================================================================================================
#=======================================================================================================================



