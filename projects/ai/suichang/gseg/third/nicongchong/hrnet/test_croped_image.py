from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import keras.backend as K
from utils.metrics import iou
from utils.loss import ce_jaccard_loss
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test(model_path, img_path1, img_path2):
    model = load_model(model_path, custom_objects={'iou': iou, 'ce_jaccard_loss': ce_jaccard_loss})

    original_image1 = tifffile.imread(img_path1)
    image1 = original_image1 / 255.0
    image1 = np.expand_dims(image1, axis=0)

    original_image2 = tifffile.imread(img_path2)
    image2 = original_image2 / 255.0
    image2 = np.expand_dims(image2, axis=0)
    original_images = [original_image1, original_image2]
    images = np.concatenate((image1, image2), axis=0)

    soft_pred = model.predict(images)
    soft_pred = [np.squeeze(soft_pred[i]) for i in range(len(soft_pred))]

    return original_images, soft_pred


original_image, soft_pred = test(model_path='seg_hrnet-08-4.2117-0.9428-0.4832.hdf5',
                                 img_path1='E:/ncc/DATA/AerialImageDataset/data/val/images/8.tif',
                                 img_path2='E:/ncc/DATA/AerialImageDataset/data/val/images/10.tif')


hard_pred = [np.where(soft_pred[i] > 0.5, 1, 0) for i in range(len(soft_pred))]

plt.subplot(231)
plt.imshow(original_image[0])
plt.subplot(232)
plt.imshow(soft_pred[0])
plt.subplot(233)
plt.imshow(hard_pred[0])
plt.subplot(234)
plt.imshow(original_image[1])
plt.subplot(235)
plt.imshow(soft_pred[1])
plt.subplot(236)
plt.imshow(hard_pred[1])
plt.show()