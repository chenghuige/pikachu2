import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import glob
np.random.seed(7)
tf.set_random_seed(7)


'''
Utilities to implement:
- Make it work with your code using the tf.Dataset API: (you will have to creathe the mask for every image label)
https://www.tensorflow.org/tutorials/eager/eager_basics#datasets
https://www.tensorflow.org/performance/datasets_performance
https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb

- Keep in mind the labels have to be with tf.keras.utils.to_categorical

- Keep in mind you have to be able to specify the image resolution and if it's a rgb or grayscale image

- Add image augmentation (image, labels and mask at the same time): 
horizontal and flips, rotations, scale and crops, horizontal and vertical shifts and intensity variance (contrast normalization, brigthness)
- Add median frequency balancing option

- Once the Dataset API is integrated, move into the TFrecords data format
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
'''

class Loader:
    def __init__(self, dataFolderPath, width=224, height=224, channels=3, n_classes=21, median_frequency=0):
        self.dataFolderPath = dataFolderPath
        self.height = height
        self.width = width
        self.channels = channels
        self.n_classes = n_classes
        self.median_frequency = median_frequency  

        print('Reading files...')

        # Load filepaths
        files = glob.glob(os.path.join(dataFolderPath, '*', '*', '*'))

        print('Structuring test and train files...')
        self.test_list = [file for file in files if '/test/' in file]
        self.train_list = [file for file in files if '/train/' in file]


        self.image_train_list = [file for file in self.train_list if '/images/' in file]
        self.image_test_list = [file for file in self.test_list if '/images/' in file]
        self.label_train_list = [file for file in self.train_list if '/labels/' in file]
        self.label_test_list = [file for file in self.test_list if '/labels/' in file]

        self.label_test_list.sort()
        self.image_test_list.sort()
        self.label_train_list.sort()
        self.image_train_list.sort()

        print('Loaded ' + str(len(self.image_train_list)) + ' training samples')
        print('Loaded ' + str(len(self.image_test_list)) + ' testing samples')

        print('Dataset contains ' + str(self.n_classes) + ' classes')

if __name__ == "__main__":

    loader = Loader('./Datasets/camvid', n_classes=11, width=480, height=360)
    #x, y, mask = loader.get_batch(size=2, augmenter='segmentation')
