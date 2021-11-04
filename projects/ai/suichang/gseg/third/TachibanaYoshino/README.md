# Remote-sensing-image-semantic-segmentation-tf2
The remote sensing image semantic segmentation repository based on tf.keras includes backbone networks such as resnet, densenet, mobilenet, and segmentation networks such as deeplabv3+, pspnet, panet, and segnet.

This repository has been used to participate in the remote sensing semantic image segmentation track of the 2020 National Artificial Intelligence Competition (NAIC).  

----
## Data description    

|class|label|
| :------: | :------: |
|Water|100|
|Transportation|200|
|Building|300|
|Arable land|400|
|Grassland|500|
|Woodland|600|
|Bare soil|700|
|Others|800|

## Requirements  
- python 3.7  
- tensorflow-gpu 2.3 
- opencv-python  
- tqdm  
- numpy    
- argparse  
- matplotlib    
- Pillow      

## Usage  
### 1. Download dataset  
>    data    

### 2. Separate the validation set from the training data (optional)   
>  `python split_val_data_from_train.py`   

### 3. Complete the basic configuration of training and testing, such as data path, model path, etc.
>  Modify the config.py file  

### 4. Train  
>  `python train.py --model DeepLabV3Plus --backBone ResNet152 --lr_scheduler cosine_decay  --lr_warmup True`  
  
### 4. Download pre-trained weights  
>  [Link (TBD)](#)  

### 5. Inference  
>  `python predict.py`  

## Results 

> MIou, FWIou  
> The inference result image is in the results folder  
