# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras.callbacks import TensorBoard

def step_decay_schedule(initial_lr=1e-5, decay_factor=0.1, step_size=30):
    """
    Define custom learning rate schedule
    """
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)
    
def manual_lr(initial_lr, epoch_marker, lrs):
    """
    Define custom learning rate schedule
    """
    
    assert len(epoch_marker) == len(lrs)
    
    def schedule(epoch):
        lr = initial_lr
        i = 0
        while epoch >= epoch_marker[i]:
            i += 1
        lr = lrs[i]
        return lr
    
    return LearningRateScheduler(schedule)
    
def ignore_unknown_xentropy(ytrue, ypred):
    """
    Define custom loss function to ignore void class
    https://github.com/keras-team/keras/issues/6261
    Assuming last class is void
    """
    return (1-ytrue[:, :, :, -1])*categorical_crossentropy(ytrue, ypred)
    
def weighted_pixelwise_crossentropy(class_weights):
    def loss(y_true, y_pred):
        _EPSILON = 1e-7
        epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss
    
class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, b_size, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.
        #self.batch_size = b_size

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * self.batch_size,) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * self.batch_size,) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
              
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)