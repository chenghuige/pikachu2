'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
@LastEditTime: 2019-11-02 16:55:49
'''
import numpy as np
import random
import torch
import os
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance,ImageFilter
import numbers
import warnings
import types
from tqdm import tqdm
from skimage.morphology import remove_small_holes


Image.MAX_IMAGE_PIXELS = 100000000000

seed = 15
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True