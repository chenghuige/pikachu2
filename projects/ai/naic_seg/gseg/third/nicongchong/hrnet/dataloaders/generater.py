from keras.utils import np_utils
import numpy as np
import tifffile
from glob import glob
import os
from albumentations import (
    Compose,
    HorizontalFlip,    # 随机水平翻转
    VerticalFlip,      # 随机垂直翻转
    RandomRotate90,    # 随机90度旋转
    RandomSizedCrop    # 随机尺寸裁剪并缩放回原始大小
)


# 1. 读取数据路径
def get_data_paths(train_images_dir, test_images_dir, seed=37):
    train_images_paths = glob(os.path.join(train_images_dir, '*.tif'))
    test_images_paths = glob(os.path.join(test_images_dir, '*.tif'))

    # 随机打乱训练数据
    np.random.seed(seed)
    np.random.shuffle(train_images_paths)

    return train_images_paths, test_images_paths


# 2. 获取batch data
def batch_generator(train_images_paths, batch_size, flag):
    while 1:
        for i in range(0, len(train_images_paths), batch_size):
            idx_start = 0 if (i + batch_size) > len(train_images_paths) else i
            idx_end = idx_start + batch_size
            if flag == 'train':
                images, gts = read_train_img(train_images_paths[idx_start: idx_end])
            else:
                images, gts = read_test_img(train_images_paths[idx_start: idx_end])
            yield (images, gts)


# 3. 读取一个batch的图片并进行实时数据扩充
def read_train_img(images_paths):
    images = []
    gts = []
    for image_path in images_paths:
        gt_path = image_path.replace('images', 'gt')

        image = tifffile.imread(image_path)
        gt = tifffile.imread(gt_path)

        # 数据扩充
        h, w = image.shape[0], image.shape[1]
        aug = Compose([VerticalFlip(p=0.5),
                       RandomRotate90(p=0.5),
                       HorizontalFlip(p=0.5),
                       RandomSizedCrop(min_max_height=(128, 512), height=h, width=w, p=0.5)])

        augmented = aug(image=image, mask=gt)
        image = augmented['image']
        gt = augmented['mask']

        # 数据预处理
        image = image / 255.0
        gt_temp = gt.copy()
        gt[gt_temp == 255] = 1
        gt = np.expand_dims(gt, axis=2)
        # gt = np_utils.to_categorical(gt, num_classes=1)

        images.append(image)
        gts.append(gt)

    return np.array(images), np.array(gts)


# 4. 读取一个batch的验证图片
def read_test_img(images_paths):
    images = []
    gts = []
    for image_path in images_paths:
        gt_path = image_path.replace('images', 'gt')

        image = tifffile.imread(image_path)
        gt = tifffile.imread(gt_path)

        # 数据预处理
        image = image / 255.0
        gt_temp = gt.copy()
        gt[gt_temp == 255] = 1
        gt = np.expand_dims(gt, axis=2)
        # gt = np_utils.to_categorical(gt, num_classes=1)

        images.append(image)
        gts.append(gt)

    return np.array(images), np.array(gts)