# -- coding: utf-8 --

import os
import cv2
import numpy as np
from model.GhostNet import GhostModel
from config.config import get_config_from_json, get_test_args
import time

modelpath = 'weight/GhostNet_model.h5'
testpath = 'test_dir/'

def get_input_from_test_dir(path,size):
    imglist = os.listdir(path)
    imgs = []
    for imgname in imglist:
        img = cv2.imread(path+imgname)
        img = cv2.resize(img,(size,size))
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

def infer():

    parser = None
    config = None

    try:
        args, parser = get_test_args()
        config,_ = get_config_from_json(args.config)
    except Exception as e:
        print('[ERROR] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Attention] 参考: python infer.py -c config/ghost_config.json')
        exit(0)

    numclass = config.num_class
    size = config.size
    use_mnist_data = config.use_mnist_data

    if(use_mnist_data):
        ghost = GhostModel(numclass,size,1)
    else:
        ghost = GhostModel(numclass,size,3)

    model = ghost.model
    model.load_weights(modelpath)
    
    x = get_input_from_test_dir(testpath,size)
    t1 = time.time()
    y = model.predict(x)
    t2 = time.time()
    print(t2-t1)
    for i in range(len(y)):
        cls = np.argmax(y[i])
        print('the NO %d image is class %d ' % (i+1,cls))

if __name__ == '__main__':
    infer()
