import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import random
from skimage.transform import rotate

from tqdm import tqdm
from torch.utils import data


class CCFLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=256):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 5
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229,0.224,0.225])
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "trainval"]:
            file_list = tuple(open(root + '/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/img/' + img_name
        lbl_path = self.root + '/label/' + img_name

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        #
        lbl[lbl==255] = 0

        #random scaleSizeCrop
        use_random_crop=True
        if(use_random_crop):
            if(self.split!='val'):
                for attempt in range(100):
                    areas = img.shape[0] * img.shape[1]
                    target_area = random.uniform(0.5, 1) * areas #input:512 ,range(198,360)and resizeto 256

                    w = int(round(np.sqrt(target_area)))
                    h = int(round(np.sqrt(target_area)))

                    if w <= img.shape[1] and h <= img.shape[0]:
                        x1 = random.randint(0, img.shape[1] - w)
                        y1 = random.randint(0, img.shape[0] - h)

                        img = img[y1:y1+h,x1:x1+w]
                        lbl = lbl[y1:y1+h,x1:x1+w]
                        if(((img.shape[1],img.shape[0]) == (w, h)) and ((lbl.shape[1],lbl.shape[0]) == (w, h))):
                            break
                assert((img.shape[1],img.shape[0]) == (w, h))
                assert((lbl.shape[1],lbl.shape[0]) == (w, h))
            else:
                w, h = img.shape[1],img.shape[0]
                th, tw = self.img_size[0],self.img_size[1]
                x1 = int(round((w - tw) / 2.))
                y1 = int(round((h - th) / 2.))
                img = img[y1:y1+h,x1:x1+w]
                lbl = lbl[y1:y1+h,x1:x1+w]

         #random rotate
        if(random.random()<0.5 and self.split!='val'):
            angle = random.randint(-90,90)
            img = rotate(img,angle,mode='symmetric',preserve_range=True)
            lbl = rotate(lbl,angle,mode='symmetric',order=0,preserve_range=True)

        #random vertically flip
        if(random.random()<0.5 and self.split!='val'):
                img = np.flip(img,axis=0)
                lbl = np.flip(lbl,axis=0)
                #print "vertically flip"

        #random horizontally flip
        if(random.random()<0.5 and self.split!='val'):
                img = np.flip(img,axis=1)
                lbl = np.flip(lbl,axis=1)
                #print "horizontally flip"

        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0
        img -= self.mean
        img = img/self.std
        #NHWC -> NCWH
        img = img.transpose(2, 0, 1)


        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        

        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def get_labels(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128]])


    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/lab-xiong.jiangfeng/Projects/CCF2017/dataset/stage1&stage2-train-crf'
    dst = CCFLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=1,shuffle=False)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        plt.figure(1)
        plt.imshow(img)
        plt.show(block=False)

        plt.figure(2)
        plt.imshow(dst.decode_segmap(labels.numpy()[0]))
        plt.show()
