import random
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 类别对应
class_values = [100, 200, 300, 400, 500, 600, 700, 800]


def rotate_bound(img, angle):
    if angle == 90:
        out = Image.fromarray(img).transpose(Image.ROTATE_90)
        return np.array(out)
    if angle == 180:
        out = Image.fromarray(img).transpose(Image.ROTATE_180)
        return np.array(out)
    if angle == 270:
        out = Image.fromarray(img).transpose(Image.ROTATE_270)
        return np.array(out)


def data_augment(x, y):
    flag = random.choice([1, 2, 3, 4, 5, 6])
    if flag == 1:
        x, y = cv2.flip(x, 1), cv2.flip(y, 1)  # Horizontal mirror
    if flag == 2:
        x, y = cv2.flip(x, 0), cv2.flip(y, 0)  # Vertical mirror
    if flag == 3:
        x, y = rotate_bound(x, 90), rotate_bound(y, 90)
    if flag == 4:
        x, y = rotate_bound(x, 180), rotate_bound(y, 180)
    if flag == 5:
        x, y = rotate_bound(x, 270), rotate_bound(y, 270)
    else:
        pass
    return x, y


def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res


def get_image_array(img):
    return np.float32(img) / 127.5 - 1


def get_segmentation_array(img, nClasses):
    assert len(img.shape) == 2
    seg_labels = np.zeros((256, 256, nClasses))

    for p in class_values:
        img[img == p] = class_values.index(p)

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    seg_labels = np.reshape(seg_labels, (256, 256, nClasses))
    return seg_labels


def train_data_generator(images_path, labels_path, batch_size, num_class, use_augment):
    img_seg_pairs = get_img_label_paths(images_path, labels_path)
    pairs_number = len(img_seg_pairs)

    while True:
        X, Y = [], []
        random.shuffle(img_seg_pairs)
        for i in range(pairs_number):
            img = cv2.imread(img_seg_pairs[i][0], cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(img_seg_pairs[i][1], cv2.IMREAD_UNCHANGED)

            if use_augment:
                img, seg = data_augment(img, seg)

            X.append(get_image_array(img))
            Y.append(get_segmentation_array(seg, num_class))

            if i == pairs_number - 1:
                yield np.array(X), np.array(Y)
                X, Y = [], []

            if len(X) == batch_size:
                assert len(X) == len(Y)
                yield np.array(X), np.array(Y)
                X, Y = [], []


def val_data_generator(images_path, labels_path, batch_size, num_class):
    img_seg_pairs = get_img_label_paths(images_path, labels_path)
    pairs_number = len(img_seg_pairs)

    while True:
        X, Y = [], []
        for i in range(pairs_number):
            img = cv2.imread(img_seg_pairs[i][0], cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(img_seg_pairs[i][1], cv2.IMREAD_UNCHANGED)

            X.append(get_image_array(img))
            Y.append(get_segmentation_array(seg, num_class))

            if i == pairs_number - 1:
                yield np.array(X), np.array(Y)
                X, Y = [], []
            if len(X) == batch_size:
                assert len(X) == len(Y)
                yield np.array(X), np.array(Y)
                X, Y = [], []


def test_data_generator(images_path, batch_size):
    images = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            assert dir_entry.endswith('tif')
            images.append(os.path.join(images_path, dir_entry))
    pairs_number = len(images)

    X, R= [], []
    for i in tqdm(range(pairs_number)):
        img = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)
        X.append(get_image_array(img))
        R.append(os.path.basename(images[i]))
        if i == pairs_number - 1:
            yield np.array(X), R
            X, R = [], []
        if len(X) == batch_size:
            yield np.array(X), R
            X, R = [], []



if __name__ == "__main__":
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [3, 5, 1, 8]]
    arr = np.array(data)
    print(arr.shape, arr)
    print(arr == 1)
    print((arr == 1).astype(int))
    print(arr)
    arr[arr == 1] = 100
    print(arr)
