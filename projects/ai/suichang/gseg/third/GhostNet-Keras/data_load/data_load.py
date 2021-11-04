# -- coding: utf-8 --

from tensorflow.keras.utils import to_categorical
import cv2
import os
import numpy as np

class Dataload(object):
    def __init__(self,train_path,test_path,numclass,size):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = "data/train.npz"
        self.test_data = "data/test.npz"
        self.__numclass = numclass
        self.__size = size
    
    def train_data_exists(self,train_data):
        return os.path.exists(train_data)

    def test_data_exists(self,test_data):
        return os.path.exists(test_data)

    def train_path_exists(self,train_path):
        return os.path.exists(train_path)
  
    def test_path_exists(self,test_path):
        return os.path.exists(test_path)

    def get_train_data(self):
        if(self.train_data_exists(self.train_data)):
            print('Wait for loading train data . . . .')
            z_train = np.load(self.train_data)
            x_train = z_train['image']
            y_train = z_train['label']
            print('finished!')
            return x_train,y_train
        elif(self.train_path_exists(self.train_path)):
            x_train = []
            y_train = []
            print('Wait for reading train data . . . .')
            index = 0
            for line in open(self.train_path):
                index = index + 1
                print(index)
                line.strip()
                img_path,label = line.split()
                img = cv2.imread(img_path)
                img = cv2.resize(img,(self.__size,self.__size))
                #img = img*1.0/255
                x_train.append(img)
                y_train.append(int(label))
            y_train = to_categorical(y_train,self.__numclass)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            print('finished!')
            print('Wait for saving train data . . . .')
            np.savez('data/train.npz',image=x_train,label=y_train)
            print('finished!')
            return x_train,y_train
        else:
            raise RuntimeError('Can not load train data,please check!')

    def get_test_data(self):
        if(self.test_data_exists(self.test_data)):
            print('Wait for loading test data . . . .')
            z_test = np.load(self.test_data)
            x_test = z_test['image']
            y_test = z_test['label']
            print('finished!')
            return x_test,y_test
        elif(self.test_path_exists(self.test_path)):
            x_test = []
            y_test = []
            print('Wait for reading test data . . . .')
            index = 0
            for line in open(self.test_path):
                index = index + 1
                print(index)
                line.strip()
                img_path,label = line.split()
                img = cv2.imread(img_path)
                img = cv2.resize(img,(self.__size,self.__size))
                #img = img*1.0/255
                x_test.append(img)
                y_test.append(int(label))
            y_test = to_categorical(y_test,self.__numclass)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            print('finished!')
            print('Wait for saving test data . . . .')
            np.savez('data/test.npz',image=x_test,label=y_test)
            print('finished!')
            return x_test,y_test
        else:
            raise RuntimeError('Can not load test data,please check!')
