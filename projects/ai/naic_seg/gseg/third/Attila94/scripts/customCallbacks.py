# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:00:29 2019

@author: beheerder
"""

import cv2
import os
import numpy as np
from keras.callbacks import Callback

class OutputObserver(Callback):
    """"
    Callback to save segmentation predictions during training.
    
    # Arguments:
        data            data that should be used for prediction
        output_path     directory where epoch predictions are to be stored
        mask_colors     class colors used for visualizing predictions
        batch_size      batch size used for prediction, default 2
        tmp_interval    save interval for tmp image in batches, default 100
    """
    def __init__(self, data, output_path, mask_colors, batch_size = 2, tmp_interval = 100):
        self.epoch = 0
        self.data = data
        self.output_path = output_path
        self.mask_colors = mask_colors
        
        if isinstance(data,(list,)):
            data_len = data[0].shape[0]
            data_out = data[0]
        else:
            data_len = data.shape[0]
            data_out = data
        
        self.batch_size = np.minimum(batch_size,data_len)
        self.tmp_interval = tmp_interval

        # Save input files
        data_out[:,:,:,0] += 103.939
        data_out[:,:,:,1] += 116.779
        data_out[:,:,:,2] += 123.68
        data_out = data_out.astype('uint8')
        if data_out.shape[-1] == 2:
            data_out = np.concatenate((data_out, np.zeros((data_out.shape[:3]+(1,)))), axis=-1)
        elif data_out.shape[-1] == 1:
            data_out = np.concatenate((data_out, np.zeros((data_out.shape[:3]+(2,)))), axis=-1)
        for i in range(data_out.shape[0]):
            cv2.imwrite(os.path.join(self.output_path,'input_{}.png'.format(i)),data_out[i,:,:,:])
        
    def labelVisualize(self, y_pred):
        """
        Convert prediction to color-coded image.
        """
        x = np.argmax(y_pred, axis=-1)
        colour_codes = np.array(self.mask_colors)
        img = colour_codes[x.astype('uint8')]
        return img
        
    def on_train_begin(self, logs={}):
        y_pred = self.model.predict(self.data, batch_size=self.batch_size)
        img = self.labelVisualize(y_pred[0,:,:,:])
        cv2.imwrite(os.path.join(self.output_path,'init.png'),img[:,:,::-1])
        
    def on_batch_end(self, batch, logs={}):
        if batch % self.tmp_interval == 0:
            y_pred = self.model.predict(self.data, batch_size=self.batch_size)
            np.save(os.path.join(self.output_path,'tmp.npy'),y_pred)
            img = self.labelVisualize(y_pred[0,:,:,:])
            cv2.imwrite(os.path.join(self.output_path,'tmp.png'),img[:,:,::-1])
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        y_pred = self.model.predict(self.data, batch_size=self.batch_size)
        np.save(os.path.join(self.output_path,'epoch_{}_img.npy'.format(epoch)),y_pred)
        for i in range(y_pred.shape[0]):
            img = self.labelVisualize(y_pred[i,:,:,:])
            cv2.imwrite(os.path.join(self.output_path,'epoch_{}_img_{}.png'.format(epoch,i)),img[:,:,::-1])
    
class LossHistory(Callback):
    """
    Callback to output losses and accuracies at each batch and epoch. 
    """
    def __init__(self, batch_log_path, epoch_log_path):
        self.epoch = 0
        self.batchinit = True
        self.epochinit = True
        self.batch_log_path = batch_log_path
        self.epoch_log_path = epoch_log_path

    def on_batch_end(self, batch, logs={}):
        # Make file and print header on first call
        if self.batchinit:
            with open(self.batch_log_path, 'a') as batch_log:
                headers = ','.join(list(map(str, logs.keys())))
                batch_log.write('epoch,{}\n'.format(headers))
            self.batchinit = False
        
        with open(self.batch_log_path, 'a') as batch_log:
            log = ','.join(list(map(str, logs.values())))
            batch_log.write('{},{}\n'.format(self.epoch,log))

    def on_epoch_end(self, epoch, logs={}):
        # Make file and print header on first call
        if self.epochinit:
            with open(self.epoch_log_path, 'a') as epoch_log:
                headers = ','.join(list(map(str, logs.keys())))
                epoch_log.write('epoch,{}\n'.format(headers))
            self.epochinit = False

        self.epoch += 1
        with open(self.epoch_log_path, 'a') as epoch_log:
            log = ','.join(list(map(str, logs.values())))
            epoch_log.write('{},{}\n'.format(epoch,log))