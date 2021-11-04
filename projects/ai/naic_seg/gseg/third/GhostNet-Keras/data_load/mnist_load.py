# -- coding: utf-8 --

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

class MnistLoad(object):
    def __init__(self,numclass,size):
        self.numclass = numclass
        self.size = size
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

        self.y_train = to_categorical(self.y_train,self.numclass)
        self.y_test = to_categorical(self.y_test,self.numclass)


    def get_train_data(self):  
        temp = []
        for img in self.x_train:
            image = cv2.resize(img,(self.size,self.size))
            image = image.reshape(self.size,self.size,1)
            temp.append(image)
        self.x_train = np.array(temp)
        return self.x_train, self.y_train

 
    def get_test_data(self):
        temp = []
        for img in self.x_test:
            image = cv2.resize(img,(self.size,self.size))
            image = image.reshape(self.size,self.size,1)
            temp.append(image)
        self.x_test = np.array(temp)
        return self.x_test, self.y_test
