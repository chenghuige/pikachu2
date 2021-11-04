'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-07-03 22:24:13
'''
from argparse import ArgumentParser
from os import mkdir,path
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
# import os
# print(os.getcwd())
from config import cfg
from data.dataloader import make_inference_dataloader
from model import build_model
import torch
import cv2 as cv

def load_arg():
    parser = ArgumentParser(description="Pytorch Hand Detection Training")
    parser.add_argument("-config_file","--CONFIG_FILE",type=str,help="Path to config file")
   
    # MODEL
    parser.add_argument('-model',"--MODEL.NET_NAME",type=str,
                        help="Net to build")
    parser.add_argument('-path',"--MODEL.LOAD_PATH",type=str,
                        help="path/file of a pretrain model(state_dict)")
    parser.add_argument("-device","--MODEL.DEVICE",type=str,
                        help="cuda:x (default:cuda:0)")
    
    # TOOLS
    parser.add_argument("-image_n","--TOOLS.image_n",type=int,default=3)
    parser.add_argument("-save_path","--TOOLS.save_path",type=str)

    arg = parser.parse_args()
    return arg

def merge_from_dict(cfg,arg_dict):
    for key in arg_dict:
        if arg_dict[key] != None:
            cfg.merge_from_list([key,arg_dict[key]])
    return cfg

def create_png(cfg):
    TOOLS = cfg.TOOLS
    image_n = TOOLS.image_n
    if image_n == 1:
        zeros = TOOLS.zeros_1
    elif image_n == 2:
        zeros = TOOLS.zeros_2
    elif image_n == 3:
        zeros = TOOLS.zeros_3
    elif image_n == 4:
        zeros = TOOLS.zeros_4
    zeros = np.zeros(zeros,np.uint8)
    return zeros

def forward(dataloader,model,png):
    for (image,pos_list) in dataloader:
        topleft_x,topleft_y,buttomright_x,buttomright_y = pos_list
        if image.cpu().data.numpy().sum() == 0:
            predict = np.zeros((1024,1024))
        else:
            
            image = image.cuda()
            predict = model(image)
            predict = torch.argmax(predict.cpu()[0],0).byte().numpy()
        if(buttomright_x-topleft_x)==1024 and (buttomright_y-topleft_y)==1024:
            png[topleft_y:buttomright_y,topleft_x:buttomright_x] = predict
        else:
            png[topleft_y:buttomright_y,topleft_x:buttomright_x] = predict[0:(buttomright_y-topleft_y),0:(buttomright_x-topleft_x)]
    return png

def label_resize_save(img,output_path):
    B = img.copy()   # 蓝色通道
    B[B == 1] = 255
    B[B == 2] = 0
    B[B == 3] = 0
    B[B == 0] = 0

    G = img.copy()   # 绿色通道
    G[G == 1] = 0
    G[G == 2] = 255
    G[G == 3] = 0
    G[G == 0] = 0

    R = img.copy()   # 红色通道
    R[R == 1] = 0
    R[R == 2] = 0
    R[R == 3] = 255
    R[R == 0] = 0
    anno_vis = np.dstack((B,G,R))
    anno_vis = cv.resize(anno_vis, None, fx= 0.1, fy=0.1)
    cv.imwrite(output_path,anno_vis)

if __name__ == "__main__":
    #0 config
    arg = load_arg()

    if arg.CONFIG_FILE != None:
        cfg.merge_from_file(arg.CONFIG_FILE)

    cfg = merge_from_dict(cfg,vars(arg))
    print(cfg)

    if cfg.OUTPUT.DIR_NAME and not path.exists(cfg.OUTPUT.DIR_NAME):
        mkdir(cfg.OUTPUT.DIR_NAME)

    #1 dataloader
    dataloader = make_inference_dataloader(cfg=cfg)

    #2 create png
    zeros = create_png(cfg)

    #3 build model
    model = build_model(cfg).cuda()
    model.eval()

    #4 forward & save
    save_path = cfg.TOOLS.save_path
    image = forward(dataloader,model,zeros)
    # cv.imwrite(save_path,image)
    pil_image = Image.fromarray(image)
    pil_image.save(save_path)
    label_save_path = path.join(path.split(save_path)[0], "vis_" + path.split(save_path)[1])
    label_resize_save(image,label_save_path)
    

    