'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
<<<<<<< HEAD
LastEditTime: 2019-06-26 10:16:59
'''

import numpy as np
import cv2 as cv
import torch

=======
LastEditTime: 2019-08-26 17:52:48
'''

from importer import *
>>>>>>> 37914f6... dockerV5_lin_modify

def _is_tensor_image(image):
    '''
    Description:  Return whether image is torch.tensor and the number of dimensions of image.
    Reutrn : True or False.
    '''
    return torch.is_tensor(image) and image.ndimension()==3

def _is_numpy_image(image):
    '''
    Description: Return whether image is np.ndarray and the number of dimensions of image
    Return: True or False.
    '''
    return isinstance(image,np.ndarray) and (image.ndim in {2,3} )

def _is_numpy(landmarks):
    '''
    Description: Return whether landmarks is np.ndarray.
    Return: True or False
    '''
    return isinstance(landmarks,np.ndarray)

def to_tensor(sample):
    '''
    Description: Convert ndarray in sample to Tensor.
    Args (type): 
        sample (np.ndarray or dict): 
            sample(ndarray):Classification
            sample({"image":image,"landmarks":landmarks}):Detection
            sample({"image":image,"mask":mask}):Segmentation
    Return: Converted sample
    '''
    #1 classification
    if isinstance(sample,np.ndarray):#Classification
        if not(_is_numpy_image(sample)):
            raise TypeError("sample should be numpy.ndarray or dict. Got {}".format(type(sample)))
        
        # handle numpy.array
        if sample.ndim == 2:
            sample = sample[:,:,None]

        # Swap color axis because 
        # numpy image: H x W x C
        # torch image: C x H x W 
        image = torch.from_numpy(sample.transpose((2,0,1)))
        if isinstance(image,torch.ByteTensor) or image.dtype == torch.uint8:
            image = image.float().div(255)
        return image

    #2 detection
    elif(('image' in sample) and ("landmarks" in sample)): 
        image,landmarks = sample["image"],sample["landmarks"]
        if not(_is_numpy_image(image)):
            raise TypeError("image should be numpy.ndarray. Got {}".format(type(image)))
        if not(_is_numpy(landmarks)):
            raise TypeError("\'landmarks\' in sample should be numpy.ndarray. Got {}".format(type(landmarks)))
        
        if image.ndim == 2:
            image = image[:,:,None]
        image = torch.from_numpy(image.transpose((2,0,1)))
        landmarks = torch.from_numpy(landmarks)
        if isinstance(image,torch.ByteTensor) or image.dtype == torch.uint8:
            image = image.float().div(255)
        landmarks = landmarks.float()   #torch.float64-->torch.float32
        sample = {'image':image,'landmarks':landmarks}
        return sample
    
    #3 segmentation
    elif(("image" in sample) and ("mask" in sample)):
        image,mask = sample["image"],sample["mask"]
        if not(_is_numpy_image(image)):
            raise TypeError("image should be numpy.ndarray. Got {}".format(type(image)))
        if not(_is_numpy_image(mask)):
            raise TypeError("image should be numpy.ndarray. Got {}".format(type(mask)))
        
        if image.ndim == 2:
            image = image[:,:,None]
        if mask.ndim == 2:
            mask = mask[:,:,None]
        image = torch.from_numpy(image.transpose((2,0,1)))
        mask  = torch.from_numpy( mask.transpose((2,0,1)))
        if isinstance(image,torch.ByteTensor) or image.dtype == torch.uint8:
            image = image.float().div(255)
        mask = mask.float()
        sample = {"image":image,"mask":mask}
        return sample
    
    else:
        raise TypeError("sample should be a dict with keys image and landmarks/mask. Got {},{}".format(sample.keys()))

def normalize(sample,mean,std,inplace=False):
    '''
    Description: Normalize a tensor image with mean and standard deviation.
    Args (type): 
        sample (torch.Tensor or dict): 
            sample(torch.Tensor):Classification
            sample({"image":image,"landmarks":landmarks}):Detection
            sample({"image":image,"mask":mask}):Segmentation
        mean (sequnence): Sequence of means for each channel.
        std (sequence): Sequence of standard devication for each channel.
    Return: 
        Converted sample
    '''
    #1 classification
    if _is_tensor_image(sample):
        if not inplace:
            sample = sample.clone()
        # check dtype and device 
        dtype = sample.dtype
        device = sample.device
        mean = torch.as_tensor(mean,dtype=dtype,device=device)
        std = torch.as_tensor(std,dtype=dtype,device=device)
        sample.sub_(mean[:,None,None]).div_(std[:,None,None])
        return sample
    #2 detection
    elif(("image" in sample) and ("landmarks" in sample)):
        image,landmarks = sample["image"],sample["landmarks"]
        if not inplace:
            image = image.clone()
         
        #check dtype and device
        dtype = image.dtype
        device = image.device
        mean = torch.as_tensor(mean,dtype=dtype,device=device)
        std = torch.as_tensor(std,dtype=dtype,device=device)
        image = image.sub_(mean[:,None,None]).div_(std[:,None,None])
        sample = {"image":image,"landmarks":landmarks}
        return sample
        
    #3 segmentation
    elif(("image" in sample) and ("mask" in sample)):
        image,mask = sample["image"],sample["mask"]
        if not inplace:
            image = image.clone()
        
        #check dtype and decice
        dtype = image.dtype
        device = image.device
        mean = torch.as_tensor(mean,dtype=dtype,device=device)
        std = torch.as_tensor(std,dtype=dtype,device=device)
        image.sub_(mean[:,None,None]).div_(std[:,None,None])
        sample = {"image":image,"mask":mask}
        return sample

    else:
        raise TypeError("sample should be a torch image or a dict with keys image and landmarks/mask. Got {},{}".format(sample.keys()))


def hflip(sample):
    '''
    Description: Horizontally flip the given sample
    Args (type): 
        sample (np.ndarray or dict):
            sample (np.ndarray): Classification.
            sample ({"image":image,"landmarks":landmarks}):Detection.
            sample ({"image":image,"mask":mask})
    Return: 
        Converted sample
    '''
    if isinstance(sample,np.ndarray):
        if _is_numpy_image(sample):
            if sample.ndim == 2:
                sample = sample[:,:,None]
            sample = cv.flip(sample,1)
            return sample
        else:
            raise TypeError("sample should be np.ndarray image. Got {}".format(type(sample)))
    elif (("image" in sample) and ("landmarks" in sample)):
        image,landmarks = sample["image"],sample["landmarks"]
        if not _is_numpy_image(image):
            raise TypeError("image should be a np.ndarray image. Got {}".format(type(image)))
        if not _is_numpy(landmarks):
            raise TypeError("landmarks should be a np.ndarray. Got {}".format(type(landmarks)))
            
        if image.ndim == 2:
            image = image[:,:,None]
        
        if image.shape[2] == 1:
            image = cv.flip(image,1)[:,:,np.newaxis] #keep image.shape = H x W x 1
        else:
            image = cv.flip(image,1)
        for index in range(len(landmarks)):
            if index%2==0:
                landmarks[index] = 1 - landmarks[index]
        sample = {"image":image,"landmarks":landmarks}
        return sample
        
    elif ("image" in sample) and ("mask" in sample):
        image,mask = sample["image"],sample["mask"]
        if not _is_numpy_image(image):
            raise TypeError("image should be a np.ndarray image. Got {}".format(type(image)))
        if not _is_numpy_image(mask):
            raise TypeError("landmarks should be a np.ndarray image. Got {}".format(type(landmarks)))
        
        if image.ndim == 2:
            image = image[:,:,None]
        # if mask.ndim == 2:    #mask要不要拉升
            # mask = mask[:,:,None]

        if image.shape[2] == 1:
            image = cv.flip(image,1)[:,:,np.newaxis]
        else:
            image = cv.flip(image,1)
        mask = cv.flip(mask,1)
        sample = {"image":image,"mask":mask}
        return sample
    else:
        raise TypeError("sample should be a torch image or a dict with keys image and landmarks/mask. Got {},{}".format(sample.keys()))
    
def vflip(sample):
    '''
    Description: Vertically flip the given sample
    Args (type): 
        sample (np.ndarray or dict):
            sample (np.ndarray): Classification.
            sample ({"image":image,"landmarks":landmarks}):Detection.
            sample ({"image":image,"mask":mask})
    Return: 
        Converted sample
    '''
    if isinstance(sample,np.ndarray):
        if _is_numpy_image(sample):
            if sample.ndim == 2:
                sample = sample[:,:,None]
            sample = cv.flip(sample,0)
            return sample
        else:
            raise TypeError("sample should be np.ndarray image. Got {}".format(type(sample)))
    elif (("image" in sample) and ("landmarks" in sample)):
        image,landmarks = sample["image"],sample["landmarks"]
        if not _is_numpy_image(image):
            raise TypeError("image should be a np.ndarray image. Got {}".format(type(image)))
        if not _is_numpy(landmarks):
            raise TypeError("landmarks should be a np.ndarray. Got {}".format(type(landmarks)))
            
        if image.ndim == 2:
            image = image[:,:,None]
        
        if image.shape[2] == 1:
            image = cv.flip(image,0)[:,:,np.newaxis] #keep image.shape = H x W x 1
        else:
            image = cv.flip(image,0)
        for index in range(len(landmarks)):
            if index%2==1:
                landmarks[index] = 1 - landmarks[index]
        sample = {"image":image,"landmarks":landmarks}
        return sample
        
    elif ("image" in sample) and ("mask" in sample):
        image,mask = sample["image"],sample["mask"]
        if not _is_numpy_image(image):
            raise TypeError("image should be a np.ndarray image. Got {}".format(type(image)))
        if not _is_numpy_image(mask):
            raise TypeError("landmarks should be a np.ndarray image. Got {}".format(type(landmarks)))
        
        if image.ndim == 2:
            image = image[:,:,None]
        if mask.ndim == 2:    #mask拉升
            mask = mask[:,:,None]

        if image.shape[2] == 1:
            image = cv.flip(image,0)[:,:,np.newaxis]
        else:
            image = cv.flip(image,0)
        mask = cv.flip(mask,0)
        sample = {"image":image,"mask":mask}
        return sample
    else:
        raise TypeError("sample should be a torch image or a dict with keys image and landmarks/mask. Got {},{}".format(sample.keys()))
            
        

        
<<<<<<< HEAD
=======
def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv.LUT(img, table)

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2]==1:
        return cv.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv.LUT(img,table)

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to 
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)

def randomcrop(sample,output_size):
    if _is_numpy_image(sample):
        h,w = sample.shape[:2]
        new_w,new_h = output_size

        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)
        sample = sample[top:top+new_h,left:left+new_w]
        return sample

    elif ("image" in sample) and ("landmarks" in sample):
        image,landmarks = sample['image'],sample['landmarks']
        h,w = image.shape[:2]
        new_w,new_h = output_size
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)

        image = image[top:top+new_h,left:left+new_w]
        landmarks = landmarks.reshape(-1,2)
        landmarks = landmarks.dot(np.array([[w,0],[0,h]]))
        landmarks = landmarks - np.array([left,top])
        landmarks = landmarks.dot(np.array([[1.0/new_w,0],[0,1.0/new_h]]))
        landmarks = landmarks.reshape(-1)

        sample = {"image":image,"landmarks":landmarks}
        return sample

    elif ("image" in sample) and ("mask" in sample):
        image,mask = sample['image'],sample['mask']
        h,w = image.shape[:2]
        new_w,new_h = output_size
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)

        image = image[top:top+new_h,left:left+new_w]
        mask = mask[top:top+new_h,left:left+new_w]
        sample = {"image":image,"mask":mask}
        return sample

    else:
        raise TypeError("sample should be a torch image or a dict with keys image and landmarks/mask. Got {},{}".format(sample.keys()))


def random_erasing(sample,sl,sh,rl,rh):
    '''
    Description: 
    Args (type): 
        p: The probability that the operation will be performed.
        sl : min erasing area.
        sh : max erasing area.
        rl : min aspect ratio.
        rh : max aspect ratio.

    Return: 
    '''
    if _is_numpy_image(sample):
        for attempt in range(20):
            h,w = sample.shape[:2]
            area = h * w
            target_area = np.random.uniform(sl,sh)*area
            aspect_ratio = np.random.uniform(rl,rh)

            RE_h = int(round(np.sqrt(target_area * aspect_ratio)))
            RE_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if RE_h < h and RE_w < w:
                x1 = np.random.randint(0,h-RE_h)
                y1 = np.random.randint(0,w-RE_w)
                sample[x1:x1+RE_h,y1:y1+RE_w,0] = np.random.randint(0,255)
                sample[x1:x1+RE_h,y1:y1+RE_w,1] = np.random.randint(0,255)
                sample[x1:x1+RE_h,y1:y1+RE_w,2] = np.random.randint(0,255)
                return sample
        return sample

    elif ("image" in sample) and ("landmarks" in sample):
        image,landmarks = sample['image'],sample['landmarks']
        for attempt in range(20):
            h,w = image.shape[:2]
            area = h * w
            target_area = np.random.uniform(sl,sh)*area
            aspect_ratio = np.random.uniform(rl,rh)

            RE_h = int(round(np.sqrt(target_area * aspect_ratio)))
            RE_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if RE_h < h and RE_w < w:
                x1 = np.random.randint(0,h-RE_h)
                y1 = np.random.randint(0,w-RE_w)
                image[x1:x1+RE_h,y1:y1+RE_w,0] = np.random.randint(0,255)
                image[x1:x1+RE_h,y1:y1+RE_w,1] = np.random.randint(0,255)
                image[x1:x1+RE_h,y1:y1+RE_w,2] = np.random.randint(0,255)
            
                sample = {'image':image,'landmarks':landmarks}
                return sample
        return sample
    elif("image" in sample) and ("mask" in sample):
        image,mask = sample['image'],sample['mask']
        for attempt in range(20):
            h,w = image.shape[:2]
            area = h * w
            target_area = np.random.uniform(sl,sh)*area
            aspect_ratio = np.random.uniform(rl,rh)

            RE_h = int(round(np.sqrt(target_area * aspect_ratio)))
            RE_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if RE_h < h and RE_w < w:
                x1 = np.random.randint(0,h-RE_h)
                y1 = np.random.randint(0,w-RE_w)
                constant = np.random.randint(100,255)
                image[x1:x1+RE_h,y1:y1+RE_w,0] = constant
                image[x1:x1+RE_h,y1:y1+RE_w,1] = constant
                image[x1:x1+RE_h,y1:y1+RE_w,2] = constant
                mask.setflags(write=1)
                mask[x1:x1+RE_h,y1:y1+RE_w] = 0
                sample = {'image':image,'mask':mask}
                return sample
        return sample

def shift_padding(sample,hor_shift_ratio,ver_shift_ratio,pad):
    image,landmarks,mask = None,None,None
    if _is_numpy_image(sample):
        image = sample
    elif ("image" in sample) and ("landmarks" in sample):
        image,landmarks = sample["image"],sample["landmarks"]
    elif ("image" in sample) and ("mask" in sample):
        image,mask = sample["image"],sample["mask"]
    else:
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    h,w = image.shape[:2]
    new_h = h + np.int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h))
    new_w = w + np.int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w))
    if image.ndim == 2:
        new_image = np.zeros((new_h,new_w),dtype=image.dtype)
    else:
        new_image = np.zeros((new_h,new_w,image.shape[-1]),dtype=image.dtype)

    new_image[int(np.round(ver_shift_ratio[1]*h)):int(np.round(ver_shift_ratio[1]*h))+h,int(np.round(hor_shift_ratio[1]*w)):int(np.round(hor_shift_ratio[1]*w))+w] = image
    top = np.random.randint(0,int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h)))
    left = np.random.randint(0,int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w)))
    image = new_image[top:top+h,left:left+w]

    if _is_numpy(landmarks):
        landmarks = landmarks.reshape(-1,2)
        landmarks = landmarks.dot(np.array([[w,0],[0,h]]))
        landmarks = landmarks + np.array([int(np.round(hor_shift_ratio[1]*w)),int(np.round(ver_shift_ratio[1]*h))])
        landmarks = landmarks - np.array([left,top])
        landmarks = landmarks.dot(np.array([[1.0/image.shape[1],0],[0,1.0/image.shape[0]]]))
        landmarks = landmarks.reshape(-1)
        return {"image":image,"landmarks":landmarks}
    if _is_numpy_image(mask):
        new_mask = np.zeros((new_h,new_w))
        new_mask[int(np.round(ver_shift_ratio[1]*h)):int(np.round(ver_shift_ratio[1]*h))+h,int(np.round(hor_shift_ratio[1]*w)):int(np.round(hor_shift_ratio[1]*w))+w] = mask
        mask = new_mask[top:top+h,left:left+w]
        return {"image":image,"mask":mask}
    return image


def gaussianblur(sample,radius):
    image,mask = sample['image'],sample['mask']
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.GaussianBlur)
    image = np.asarray(image)
    return {"image":image,"mask":mask}
>>>>>>> 37914f6... dockerV5_lin_modify
