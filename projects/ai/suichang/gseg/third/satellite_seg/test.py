import sys
import torch
import visdom
import argparse
import numpy as np
import scipy.misc as misc
import torchvision.models as models
import os

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from utils.metrics import scores
from utils.preprocess import segmap
from torch.nn import DataParallel
from models import get_model

import skimage
from skimage import transform

replace_background = False
exclude_background = False

def img_transform(img,input_size):
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img = misc.imresize(img, (input_size, input_size))
    img = img.astype(float) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229,0.224,0.225])
    img -= mean
    img = img/std
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img

def process_batch(patches,model,labels_per_pixel_list,new_w):
    images = patches[0]
    images = Variable(images.cuda())

    outputs = model(images)
    if(isinstance(outputs,tuple)):
        outputs = outputs[0]

    if(exclude_background):
        pred = outputs.data[:,1:,:,:].max(1)[1].cpu().numpy() + 1
    else:
        pred = outputs.data.max(1)[1].cpu().numpy()
    


    for i in range(len(patches[0])):
        x1 = patches[1][i][0]
        x2 = patches[1][i][1]
        y1 = patches[1][i][2]
        y2 = patches[1][i][3]

        if(pred[i].shape[0]!=(y2-y1)):
            resize_map = transform.resize(pred[i],(y2-y1,x2-x1),order=0,preserve_range=True)
        else:
            resize_map = pred[i]


        for pix_inx in range(x1,x2):
            for pix_iny in  range(y1,y2):
                location = pix_inx + pix_iny * new_w
                pred_label = resize_map[pix_iny-y1][pix_inx-x1]
                labels_per_pixel_list[location].append(pred_label)
    return 
def process_single_scale(args,model,crop_scale):
    #return a map for each scale
    batch_size = args.batch_size
    stride = args.stride

    print "Read Input Image from : {}".format(args.img_path)
    print "Processing scale: ", crop_scale
    img = misc.imread(args.img_path)
    n_classes = 5
    h,w = img.shape[0],img.shape[1]

    new_h = int(np.ceil((h-crop_scale)*1.0/stride)*stride)+crop_scale
    new_w = int(np.ceil((w-crop_scale)*1.0/stride)*stride)+crop_scale
    labels_per_pixel_list=[[] for i in range(new_w*new_h)]

    pad_image = np.zeros((new_h,new_w,3),np.uint8)
    num_cols = (new_w-crop_scale)/stride + 1
    num_rows = (new_h-crop_scale)/stride + 1
    pad_image[:h,:w] = img.copy()

    batch_index = 0
    
    patches=[[],[]]

    for i in tqdm(range(num_rows)):
        x1=0
        y1 = i*stride
        y2 = y1 + crop_scale
        for j in range(num_cols):
            x1 = j*stride
            x2 = x1 + crop_scale

            patch=pad_image[y1:y2,x1:x2]
            patch = img_transform(patch,args.input_size)
            if(len(patches[1])==0):
                patches[0]=patch
                patches[1]=[[x1,x2,y1,y2]]
            else:
                patches[0] = torch.cat((patches[0],patch),dim=0)
                patches[1].append([x1,x2,y1,y2])
            batch_index = batch_index + 1

            is_last_patch = (i==(num_rows-1) and j==(num_cols-1))

            if(batch_index%(batch_size)==0 or is_last_patch):
                process_batch(patches,model,labels_per_pixel_list,new_w)
                patches=[[],[]]

    for index in tqdm(range(new_w*new_h)):
        most_count_label = np.argmax(np.bincount(labels_per_pixel_list[index]))
        if(replace_background):
            if(most_count_label==0):
                most_count_label = 1 #plant

        labels_per_pixel_list[index] = most_count_label

    pred_labels=np.reshape(labels_per_pixel_list,(new_h,new_w))[:h,:w]

    return pred_labels

def test_large_img(args):
    # Setup Model
    #model = torch.load(args.model_path,map_location=lambda storage,loc: storage)
    #model = torch.load(args.model_path)
    #load model from model files(mode train on DataParallel)
    model = get_model(args.model_path.split('/')[-2], 5)
    state_dict = torch.load(args.model_path).state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name =k[7:] #remove moudle
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = DataParallel(model.cuda(),device_ids=[i for i in range(len(args.gpu))])
    
    model.cuda()
    model.eval()

    pred_labels_list=[]
    for crop_scale in args.crop_scales:
        pred_labels_single = process_single_scale(args,model,crop_scale)
        pred_labels_list.append(pred_labels_single)
        color_mask = segmap(pred_labels_single)
        test_id = os.path.basename(args.img_path)[0]
        misc.imsave(os.path.join(args.tempdir,"%s_temp_scale_%d.png"%(test_id,crop_scale)),color_mask)

    if(len(args.crop_scales)==1):
        pred_labels = pred_labels_list[0]
    else:
        average_map = np.zeros_like(pred_labels_list[0])
        for i in range(average_map.shape[0]):
            for j in range(average_map.shape[1]):
                pre_list=[]
                for index in range(len(args.crop_scales)):
                    pre_list.append(pred_labels_list[index][i][j])
                most_label = np.argmax(np.bincount(pre_list))
                average_map[i][j] = most_label
        pred_labels = average_map

    misc.imsave(args.out_path,np.asarray(pred_labels,dtype=np.uint8))
    color_mask = segmap(pred_labels)
    misc.imsave(args.vis_out_path,color_mask)

    if(args.img_label_path!=None):
        gts = misc.imread(args.img_label_path)
        score, class_iou = scores(gts, pred_labels, n_class=n_classes)

        for k, v in score.items():
            print k, v

        for i in range(n_classes):
            print i, class_iou[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default=None, 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    parser.add_argument('--vis_out_path', nargs='?', type=str, default=None, 
                        help='visulization of the output segmap')
    parser.add_argument('--img_label_path',nargs='?', type=str, default=None, 
                        help='Segmentation label')
    parser.add_argument('--gpu',nargs='*', type=int, default=0,
                        help='GPUIDS')
    parser.add_argument('--stride',nargs='?', type=int, default=50, 
                        help='stride of crop')
    parser.add_argument('--batch_size',nargs='?', type=int, default=32, 
                        help='batch_size of test images')
    parser.add_argument('--crop_scales',nargs='*', type=int, default=[192,224,256],
                        help='crop_scales of input image')
    parser.add_argument('--input_size',nargs='?', type=int, default=224,
                        help='input_size of network')
    parser.add_argument('--tempdir',nargs='?', type=str, default='.',
                        help='temp results')

    args = parser.parse_args()
    test_large_img(args)
