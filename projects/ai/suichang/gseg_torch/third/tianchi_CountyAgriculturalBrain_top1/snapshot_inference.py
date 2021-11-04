'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 使用膨胀预测的输出实验结果
LastEditTime: 2019-09-17 14:51:59
'''
from argparse import ArgumentParser
from os import mkdir,path
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
import os
# print(os.getcwd())
from config import cfg
from data.dataloader import make_inference_dataloader
from model import build_model
import torch
import torch.nn as nn
import cv2 as cv
from tqdm import tqdm
from glob import glob
Image.MAX_IMAGE_PIXELS = 1000000000000000000

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
    parser.add_argument("-save_path","--TOOLS.save_path",type=str,required=True)
    arg = parser.parse_args()
    return arg

def merge_from_dict(cfg,arg_dict):
    for key in arg_dict:
        if arg_dict[key] != None:
            cfg.merge_from_list([key,arg_dict[key]])
    return cfg

def get_shape(cfg):
    TOOLS = cfg.TOOLS
    image_n = TOOLS.image_n
    if image_n == 3:
        zeros = TOOLS.zeros_3
    elif image_n == 4:
        zeros = TOOLS.zeros_4
    elif image_n == 5:
        zeros = TOOLS.zeros_5
    elif image_n == 6:
        zeros = TOOLS.zeros_6
    return zeros

def create_png(cfg):
    zeros = get_shape(cfg)
    h,w = zeros[0],zeros[1]
    new_h,new_w = (h//1024+1)*1024,(w//1024+1)*1024 #填充右边界
    zeros = (new_h+512,new_w+1024)  #填充空白边界

    zeros = np.zeros(zeros,np.uint8)
    return zeros

def tta_forward(cfg,dataloader,model,png):
    device = cfg.MODEL.DEVICE
    with torch.no_grad():
        for (image,pos_list) in tqdm(dataloader):
            # forward --> predict
            image = image.cuda(device) # 复制image到model所在device上

            predict_1 = model(image)

            predict_2 = model(torch.flip(image,[-1]))
            predict_2 = torch.flip(predict_2,[-1])

            predict_3 = model(torch.flip(image,[-2]))
            predict_3 = torch.flip(predict_3,[-2])

            predict_4 = model(torch.flip(image,[-1,-2]))
            predict_4 = torch.flip(predict_4,[-1,-2])

            predict_list = predict_1 + predict_2 + predict_3 + predict_4   
            predict_list = torch.argmax(predict_list.cpu(),1).byte().numpy() # n x h x w
        
            batch_size = predict_list.shape[0] # batch大小
            for i in range(batch_size):
                predict = predict_list[i]
                pos = pos_list[i,:]
                [topleft_x,topleft_y,buttomright_x,buttomright_y] = pos

                if(buttomright_x-topleft_x)==1024 and (buttomright_y-topleft_y)==1024:
                    png[topleft_y+256:buttomright_y-256,topleft_x+256:buttomright_x-256] = predict[256:768,256:768]
                else:
                    raise ValueError("target_size!=512， Got {},{}".format(buttomright_x-topleft_x,buttomright_y-topleft_y))
    
    h,w = png.shape
    png =  png[256:h-256,256:w-256] # 去除整体外边界
    zeros = get_shape(cfg)  # 去除补全512整数倍时的右下边界
    png = png[:zeros[0],:zeros[1]]
    return png

def label_resize_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签 
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    label = cv.resize(label.copy(),None,fx=0.1,fy=0.1)
    r = np.where(label == 1, 255, 0)
    g = np.where(label == 2, 255, 0)
    b = np.where(label == 3, 255, 0)
    yellow = np.where(label == 4, 255, 0)
    anno_vis = np.dstack((b, g, r)).astype(np.uint8)
    # 黄色分量(红255, 绿255, 蓝0)
    anno_vis[:, :, 0] = anno_vis[:, :, 0] + yellow
    anno_vis[:, :, 1] = anno_vis[:, :, 1] + yellow
    anno_vis[:, :, 2] = anno_vis[:, :, 2] + yellow
    if img is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(img, alpha, anno_vis, 1-alpha, 0)
        return overlapping

def snapshot_forward(cfg,dataloader,model_list,png):
    device = cfg.MODEL.DEVICE
    with torch.no_grad():
        for (image,pos_list) in tqdm(dataloader):
            # forward --> predict
            image = image.cuda(device) # 复制image到model所在device上
            predict_list = 0
            for model in model_list:
                predict_1 = model(image)
                predict_list = predict_1
                predict_2 = model(torch.flip(image,[-1]))
                predict_2 = torch.flip(predict_2,[-1])

                predict_3 = model(torch.flip(image,[-2]))
                predict_3 = torch.flip(predict_3,[-2])

                predict_4 = model(torch.flip(image,[-1,-2]))
                predict_4 = torch.flip(predict_4,[-1,-2])

                predict_list += (predict_1 + predict_2 + predict_3 + predict_4)   
            predict_list = torch.argmax(predict_list.cpu(),1).byte().numpy() # n x h x w
        
            batch_size = predict_list.shape[0] # batch大小
            for i in range(batch_size):
                predict = predict_list[i]
                pos = pos_list[i,:]
                [topleft_x,topleft_y,buttomright_x,buttomright_y] = pos

                if(buttomright_x-topleft_x)==1024 and (buttomright_y-topleft_y)==1024:
                    png[topleft_y+256:buttomright_y-256,topleft_x+256:buttomright_x-256] = predict[256:768,256:768]
                else:
                    raise ValueError("target_size!=512， Got {},{}".format(buttomright_x-topleft_x,buttomright_y-topleft_y))
    
    h,w = png.shape
    png =  png[256:h-256,256:w-256] # 去除整体外边界
    zeros = get_shape(cfg)  # 去除补全512整数倍时的右下边界
    png = png[:zeros[0],:zeros[1]]
    return png




if __name__ == "__main__":
    model_path = glob(r"./output/model/epoch_15*.pth")
    # model_path = [
        # r"./output/model/epoch_15_deeplabv3plus_resnet101_84983630.pth"
        # r"/media/pci/d766124d-ad95-474c-8464-f935fd6c426c/lhh/Project/tianchi_AI2019/code/round2_deeplabv3plus_modify/output/model/epoch_5_deeplabv3+modify_deeplabv3plus_resnet101{'pixel_error': 0.7875426444273231}.pth"
        # r"/media/pci/d766124d-ad95-474c-8464-f935fd6c426c/lhh/Project/tianchi_AI2019/code/round2_deeplabv3plus_modify/output/model/epoch_7_deeplabv3+modify_deeplabv3plus_resnet101{'pixel_error': 0.7785988135626397}.pth",
        # r"/media/pci/d766124d-ad95-474c-8464-f935fd6c426c/lhh/Project/tianchi_AI2019/code/round2_deeplabv3plus_modify/output/model/epoch_1_deeplabv3+modify_deeplabv3plus_resnet101{'pixel_error': 0.7851126939226043}.pth"
    
        # ]

    #0 config
    arg = load_arg()
    
    if arg.CONFIG_FILE != None:
        cfg.merge_from_file(arg.CONFIG_FILE)

    cfg = merge_from_dict(cfg,vars(arg))
    print(cfg)

    if cfg.OUTPUT.DIR_NAME and not path.exists(cfg.OUTPUT.DIR_NAME):
        mkdir(cfg.OUTPUT.DIR_NAME)
    save_path = cfg.TOOLS.save_path
    
    #3 build model
    torch.backends.cudnn.benchmark = True
    model_list = []
    for path in model_path:
        cfg.MODEL.LOAD_PATH = path
        model = build_model(cfg)
        # cfg.MODEL.DEVICE = "cuda:2"
        model = model.cuda(cfg.MODEL.DEVICE)
        model = nn.DataParallel(model)
        # model = nn.DataParallel(model,device_ids=[0,1,2,3])
        model.eval()
        print("build model")
        model_list.append(model)

    for i in [3,4]:
        cfg.TOOLS.image_n = i
        #1 dataloader
        dataloader = make_inference_dataloader(cfg=cfg)

        #2 create png
        zeros = create_png(cfg)


        # #4 forward & save
        image = snapshot_forward(cfg,dataloader,model_list,zeros)
        # mask = np.asarray(Image.open(r"./output/source/image_"+str(i)+"_mask.png"))
        # image = image*mask
        pil_image = Image.fromarray(image)
        pil_image.save("image_"+str(i)+"_predict.png")
        label_save_path = "vis_image_"+str(i)+"_predict.jpg"
        source = cv.imread(r"./output/source/image_"+str(i)+"_vis.png")
        overlap = label_resize_vis(image.copy(),source)
        cv.imwrite(label_save_path,overlap)

        label_save_path = "ori_vis_image_"+str(i)+"_predict.jpg"
        overlap = label_resize_vis(image.copy(),None)
        cv.imwrite(label_save_path,overlap)
    

    