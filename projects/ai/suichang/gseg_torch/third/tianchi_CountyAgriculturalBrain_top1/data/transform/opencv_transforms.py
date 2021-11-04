'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
<<<<<<< HEAD
LastEditTime: 2019-07-02 11:27:25
=======
LastEditTime: 2019-08-26 17:52:42
>>>>>>> 37914f6... dockerV5_lin_modify
'''



from . import opencv_functional as Fun
from importer import *
__all__ = []

class GaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self,p=0.2,radiu=2):
        self.p = p
        self.radiu = 2
    def __call__(self,sample):
        if np.random.uniform()<self.p:
            return Fun.gaussianblur(sample,self.radiu)
        else:
            return sample

class Shift_Padding(object):
    def __init__(self,p=0.1,hor_shift_ratio=0.1,ver_shift_ratio=0.1,pad=0):
        assert isinstance(hor_shift_ratio,(float,list,tuple))
        if isinstance(hor_shift_ratio,float):
            self.hor_shift_ratio = (-hor_shift_ratio,hor_shift_ratio)
        else:
            assert len(hor_shift_ratio) == 2
            self.hor_shift_ratio = hor_shift_ratio

        assert isinstance(ver_shift_ratio,(float,list,tuple))
        if isinstance(ver_shift_ratio,float):
            self.ver_shift_ratio = (-ver_shift_ratio,ver_shift_ratio)
        else:
            assert len(ver_shift_ratio) == 2
            self.ver_shift_ratio = ver_shift_ratio
        self.pad = pad
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return Fun.shift_padding(sample,hor_shift_ratio=self.hor_shift_ratio,ver_shift_ratio=self.ver_shift_ratio,pad=self.pad)
        else:
            return sample
    def __repr__(self):
        return __class__.__name__ + "(p={},hor_shift_ratio={},ver_shift_ratio={},pad={})".format(self.p,self.hor_shift_ratio,self.ver_shift_ratio,self.pad)




class RandomChoice(object):
    """
    Apply transformations randomly picked from a list with a given probability
    Args:
        transforms: a list of transformations
        p: probability
    """
    def __init__(self,p,transforms):
        self.p = p
        self.transforms = transforms
    def __call__(self,sample):
        if len(self.transforms) < 1:
            raise TypeError("transforms(list) should at least have one transformation")
        for t in self.transforms:
            if np.random.uniform(0,1) < self.p:
                sample = t(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)

class Compose(object):
    '''
    Description: Compose several transforms together
    Args (type): 
        transforms (list): list of transforms
        sample (ndarray or dict):
    return: 
        sample (ndarray or dict)
    '''
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    '''
    Description: Convert ndarray in sample to Tensors.
    Args (type): 
        sample (ndarray or dict)
    return: 
        Converted sample.
    '''
    def __call__(self,sample):
        return Fun.to_tensor(sample)
    def __repr__(self):
        return self.__class__.__name__ + "()"

class Normalize(object):
    '''
    Description: Normalize a tensor with mean and standard deviation.
    Args (type): 
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of std for each channel.
    Return: 
        Converted sample
    '''
    def __init__(self,mean,std,inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self,sample):
        #Convert to tensor
        mean = torch.tensor(self.mean,dtype=torch.float32)
        std = torch.tensor(self.std,dtype=torch.float32)
        return Fun.normalize(sample,mean,std,inplace=self.inplace)
    def __repr__(self):
        format_string = self.__class__.__name__ + "(mean={0},std={1})".format(self.mean,self.std)
        return format_string

class RandomHorizontalFlip(object):
    '''
    Description: Horizontally flip the given sample with a given probability.
    Args (type): 
        p (float): probability of the image being flipped. Default value is 0.5.
    Return: Converted sample
    '''
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return Fun.hflip(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    '''
    Description: Vertically flip the given sample with a given probability.
    Args (type): 
        p (float): probability of the image being flipped. Default value is 0.5.
    Return: 
        Converted sample
    '''
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return Fun.vflip(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

<<<<<<< HEAD

=======
        
class Lambda(object):
    '''
    Description: Apply a user-defined lambda as a transform.
    Args (type): lambd (function): Lambda/function to be used for transform.
    Return: 
    '''
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
>>>>>>> 37914f6... dockerV5_lin_modify

class RandomChoce(object):
    '''
    Description: Apply transformations randomly picked from a list with a given probability
    Args (type): 
        transforms: a list of transformations
        p: probability
    Return: 
        shuffle_transform_list
    '''
<<<<<<< HEAD
    def __init__(self,p,transforms):
        self.p = p
        self.transforms = transforms
    def __call__(self,sample):
        if len(self.transforms) < 1:
            raise TypeError("transforms(list) should at least have one transformation")
        for t in self.transforms:
            if np.random.uniform(0,1) < self.p:
                sample = t(sample)
        return sample
    
=======
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn('Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn('Hue jitter enabled. Will slow down loading immensely.')
    
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = np.random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: Fun.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = np.random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: Fun.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = np.random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: Fun.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = np.random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: Fun.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        if (isinstance(sample,np.ndarray) and (sample.ndim in {2,3})):
            return transform(sample)
        elif ("image" in sample and "landmarks" in sample):
            image,landmarks = sample['image'],sample["landmarks"]
            image = transform(image)
            sample = {"image": image,"landmarks": landmarks}
            return sample
        elif ("image" in sample and "mask" in sample):
            image,mask = sample["image"],sample['mask']
            image = transform(image)
            sample = {"image":image,"mask":mask}
            return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandomCrop(object):
    '''
    Description:  Crop randomly the image in a sample
    Args (type): 
        output_size(tuple or int):Desized output size.
        If int,square crop is made
    Return: 
    '''
    def __init__(self,p,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return Fun.randomcrop(sample,self.output_size)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + '(output_size={})'.format(self.output_size)
    

class RandomErasing(object):
    def __init__(self,p=0.5,sl=0.02,sh=0.4,rl=0.2):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = 1/rl

    def __call__(self,sample):
        if np.random.random() < self.p:
            return Fun.random_erasing(sample,self.sl,self.sh,self.rl,self.rh)
        else:
            return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "(sl={},sh={},rl={},rh={})".format(self.sl,self.sh,self.rl,self.rh)
>>>>>>> 37914f6... dockerV5_lin_modify
