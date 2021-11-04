'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-07-03 13:35:09
'''

from .build import PNG_Dataset,Inference_Dataset


def build_dataset(cfg,transforms=None,is_train=True):
    '''
    Description: build_dataset
    Args (type): 
        cfg (yaml): config file.
        transforms (callable,optional): Optional transforms to be applied on a sample.
        is_train (bool): True or False.
    return: 
        dataset(torch.utils.data.Dataset)
    '''
    DATASET = cfg.DATA.DATASET
    if is_train==True:
        csv_file = DATASET.train_csv_file
        image_dir = DATASET.train_root_dir
        mask_dir  = DATASET.train_mask_dir
    else:
        csv_file = DATASET.val_csv_file
        image_dir = DATASET.val_root_dir
        mask_dir = DATASET.val_mask_dir
    
    dataset = PNG_Dataset(cfg,csv_file,image_dir,mask_dir,transforms)

    return dataset


def build_inferience_dataset(root_dir,csv_file,transforms=None):
    dataset = Inference_Dataset(root_dir=root_dir,csv_file=csv_file,transforms=transforms)
    return dataset