'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 根据cfg配置文件，加载数据增强函数
<<<<<<< HEAD
LastEditTime: 2019-07-02 15:50:42
=======
LastEditTime: 2019-09-02 20:31:04
>>>>>>> 37914f6... dockerV5_lin_modify
'''
from . import opencv_transforms as transforms

def compose(list_transforms):
    '''
    Description: 
    Args (type): 
    Return: 
    '''
    add_transforms = []

    for [transform,para] in list_transforms:
        # print(transform,para)
        add_transforms.append(getattr(transforms,transform)(*para))

    return add_transforms

    
def build_transforms(cfg,is_train=True):
    '''
    Description: 
    Args (type): 
        cfg (yaml): config file.
        is_train (bool): True or False.
    return: 
        
    '''
    TRANSFORMS = cfg.DATA.TRANSFORMS

    if is_train==True:
        list_transforms = list(TRANSFORMS.TRAIN.items())
        list_transforms = compose(list_transforms)  # 转为list

        if TRANSFORMS.ENABLE_RANDOMCHOICE == True:
            p = TRANSFORMS.RandomChoce
            list_transforms = [transforms.RandomChoce(p=p,transforms=list_transforms)]
  
    else:
        list_transforms = list(TRANSFORMS.VAL.items())
        list_transforms = compose(list_transforms)
    
    list_transforms.append(transforms.ToTensor())
    if TRANSFORMS.ENABLE_NORMALIZE == True:
        para = TRANSFORMS.Normalize
        list_transforms.append(transforms.Normalize(*para))
    if TRANSFORMS.ENABLE_RANDOM_CROP:
        list_transforms = [transforms.RandomCrop(*TRANSFORMS.RandomCrop)] + list_transforms
    print(list_transforms)
    compose_transforms = transforms.Compose(list_transforms)
    return compose_transforms


def build_inference_transforms(cfg):
    '''
    Description: 
    Args (type): 
        cfg (yaml): config file.
        is_train (bool): True or False.
    return: 
        
    '''
    TRANSFORMS = cfg.DATA.TRANSFORMS

    list_transforms = list(TRANSFORMS.VAL.items())
    list_transforms = compose(list_transforms)
    
    list_transforms.append(transforms.ToTensor())
    if TRANSFORMS.ENABLE_NORMALIZE == True:
        para = TRANSFORMS.Normalize
        list_transforms.append(transforms.Normalize(*para))
    print(list_transforms)
    compose_transforms = transforms.Compose(list_transforms)
    return compose_transforms