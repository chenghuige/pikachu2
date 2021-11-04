'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
<<<<<<< HEAD
LastEditTime: 2019-07-03 17:05:17
=======
LastEditTime: 2019-09-17 14:46:15
>>>>>>> 37914f6... dockerV5_lin_modify
'''

from importer import *

class PNG_Dataset(Dataset):
    """ Tianchi AI 2019 """
    def __init__(self,cfg,csv_file,image_dir,mask_dir,transforms=None):
        '''
        Description: 
        Args (type): 
            cfg (yaml): config file.
            csv_file  (string): Path to the file with annotations.
            image_dir (string): Derectory with all images.
            mask_dir (string): Derectory with all labels.
            transforms (callable,optional): Optional transforms to be applied on a sample.
        return: 
        '''
        self.cfg = cfg
        self.csv_file = pd.read_csv(csv_file,header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self,idx):
        """
        Args:
            idx (int): index of sample
        """
        filename = self.csv_file.iloc[idx,0]
        dir,filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir,filename)
        mask_path = os.path.join(self.mask_dir,filename)
        image = Image.open(image_path)
        image = np.asarray(image) #mode:RGBA

        image = cv.cvtColor(image,cv.COLOR_RGBA2BGRA) # PIL(RGBA)-->cv2(BGRA)
<<<<<<< HEAD
=======
        image = cv.cvtColor(image,cv.COLOR_BGRA2RGB)


>>>>>>> 37914f6... dockerV5_lin_modify
        mask = np.asarray(Image.open(mask_path)) #mode:P(单通道)
        mask = mask.copy()

        sample = {'image':image,'mask':mask}

        if self.transforms:
            sample = self.transforms(sample)

        image,mask = sample['image'],sample['mask']
        # print((image.dtype),(mask.dtype))
        return image,mask


class Inference_Dataset(Dataset):
    def __init__(self,root_dir,csv_file,transforms=None):
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file,header=None)
        self.transforms = transforms
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self,idx):
        filename = self.csv_file.iloc[idx,0]
        root_dir,filename = os.path.split(filename)
        image_path = os.path.join(self.root_dir,filename)
        image = np.asarray(Image.open(image_path)) #mode:RGBA
        image = cv.cvtColor(image,cv.COLOR_RGBA2BGRA) # PIL(RGBA)-->cv2(BGRA)

        if self.transforms:
            image = self.transforms(image)

        
        pos_list = self.csv_file.iloc[idx,2:].values.astype("int")  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
        return image,pos_list

    
