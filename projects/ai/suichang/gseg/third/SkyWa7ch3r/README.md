# ImageSegmentation
This project is a part of the Pawsey Summer Internship where I will do test multiple semantic segmentation algorithms and models on their training and inference time. To do this TensorFlow 2.1 will be used benchmark them against one another primarily on the Cityscapes Dataset. The main goal is to asceertain the usefulness of these models for Critical Infrastructure like railways, buildings roads etc. Their inference times will likely be assessed on a smaller machine like the NVIDIA Jetson AGX to assess their real-time practical use. This makes Cityscapes a perfect dataset to benchmark on. Reimplementing certain models will be easier than others, while some will be difficult as they were made in PyTorch or much older versions of TF.

Semantic Segmentation or Pixel-based Labelling is the process of automatically applying a class or label, designated by a dataset, to each pixel in an image. In the case of Cityscapes, the classes could be a car, truck, building, person etc. Please do not confuse this with Instance Segmentation, where each object is treated as a separate class, so an image with multiple people will have classes object1, object2, object3 etc. although the classes are still designated by the dataset we are looking for individual objects, or **instances** of objects. Panoptic Segmentation, is the combination of both Semantic and Instance Segmentation, to find the general class, say a car, the instances of those objects, e.g. an image with multiple cars with have classes car1, car2, car3 etc.

## Notes on the restrictions of this project 
I will be wanting to train at full resolution of the cityscapes dataset which is 1024x2048px, a rather high resolution for many architectures. This will no doubt challenge the Xavier's capbilities. This will also highlight the differences between current models and the older models.

## Using train.py
<pre>
python train.py --help<br>
usage: train.py [-h] -m {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet} <br>
                [-r RESUME] [-p PATH] [-c] [-t TARGET_SIZE] [-b BATCH]<br>
                [-e EPOCHS] [--mixed-precision]<br>
                [-f {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd,sgd_nesterov}]<br>
                [--schedule {polynomial,cyclic}] [--momentum MOMENTUM]<br>
                [-l LEARNING_RATE] [--max-lr MAX_LR] [--min-lr MIN_LR]<br>
                [--power POWER] [--cycle CYCLE]<br>
<br>
Start training a semantic segmentation model<br>
<br>
optional arguments:<br>
  -h, --help            show this help message and exit<br>
  -m {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}, --model {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}<br>
                        Specify the model you wish to use: OPTIONS: unet,<br>
                        separable_unet, bayes_segnet, deeplabv3+, fastscnn<br>
  -r RESUME, --resume RESUME<br>
                        Resume the training, specify the weights file path of<br>
                        the format weights-[epoch]-[val-acc]-[val-<br>
                        loss]-[datetime].hdf5<br>
  -p PATH, --path PATH  Specify the root folder for cityscapes dataset, if not
                        used looks for CITYSCAPES_DATASET environment variable
  -c, --coarse          Use the coarse images<br>
  -t TARGET_SIZE, --target-size TARGET_SIZE<br>
                        Set the image size for training, should be a elements<br>
                        of a tuple x,y,c<br>
  -b BATCH, --batch BATCH<br>
                        Set the batch size<br>
  -e EPOCHS, --epochs EPOCHS<br>
                        Set the number of Epochs<br>
  --mixed-precision     Use Mixed Precision. WARNING: May cause memory leaks<br>
  -f {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd,sgd_nesterov}, --lrfinder <br>{adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd,sgd_nesterov}<br>
                        Use the Learning Rate Finder on a model to determine<br>
                        the best learning rate range for said optimizer<br>
  --schedule {polynomial,cyclic}<br>
                        Set a Learning Rate Schedule, here either Polynomial<br>
                        Decay (polynomial) or Cyclic Learning Rate (cyclic) is<br>
                        Available<br>
  --momentum MOMENTUM   Only useful for lrfinder, adjusts momentum of an<br>
                        optimizer, if there is that option<br>
  -l LEARNING_RATE, --learning-rate LEARNING_RATE<br>
                        Set the learning rate<br>
<br>
schedule:<br>
  Arguments for the Learning Rate Scheduler<br>
<br>
  --max-lr MAX_LR       The maximum learning rate during training<br>
  --min-lr MIN_LR       The minimum learning rate during training<br>
  --power POWER         The power used in Polynomial Decay<br>
  --cycle CYCLE         The length of cycle used for Cyclic Learning Rates<br>
</pre>
## Using Predict
<pre>
python predict.py
usage: predict [-h] -m {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}
               [-w WEIGHTS] [-p PATH] [-r RESULTS_PATH] [-c] [-t TARGET_SIZE]
               [--backbone {mobilenetv2,xception}]

Generate predictions from a batch of images from cityscapes

optional arguments:
  -h, --help            show this help message and exit
  -m {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}, --model {unet,bayes_segnet,deeplabv3+,fastscnn,separable_unet}
                        Specify the model you wish to use: OPTIONS: unet,
                        separable_unet, bayes_segnet, deeplabv3+, fastscnn
  -w WEIGHTS, --weights WEIGHTS
                        Specify the weights path
  -p PATH, --path PATH  Specify the root folder for cityscapes dataset, if not
                        used looks for CITYSCAPES_DATASET environment variable
  -r RESULTS_PATH, --results-path RESULTS_PATH
                        Specify the path for results
  -c, --coarse          Use the coarse images
  -t TARGET_SIZE, --target-size TARGET_SIZE
                        Set the image size for training, should be a elements
                        of a tuple x,y,c
  --backbone {mobilenetv2,xception}
                        The backbone for the deeplabv3+ model
</pre>                        
## Using convert_to_onnx
<pre>
python convert_to_onnx.py --help
usage: convert_to_onnx.py [-h] [-m {fastscnn,deeplabv3+,separable_unet}]
                          [-w WEIGHTS_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -m {fastscnn,deeplabv3+,separable_unet}, --model {fastscnn,deeplabv3+,separable_unet}
                        Model to convert to uff
  -w WEIGHTS_FILE, --weights-file WEIGHTS_FILE
                        The weights the model will use in the uff file
</pre>                        
### Convert_to_ONNX
Please ensure that this is converted on the Jetson Xavier for best results. Which at this time needed to use Tensorflow 1.15.0 and TensorRT 6, although I'm still trying to work through issues with using due to the restrictions on TensorFlow version and TensorRT version.

## Metrics
To understand the metrics used here, please goto: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
Some models require the use of ImageNet pretrained models to initialize their weights

## Requirements for Project
* TensorFlow 2.1 (Mainly testing out its capabilities at this time)
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores (Project will use different GPUs)
   * On the Jetson AGX Xavier its TensorRT version 6.
* Python >= 3.7

## Semantic Segmentation
The models that will be trained and tested against will be:
### U-Net 
Following this github repo here: https://github.com/zhixuhao/unet <br>
Which was inspired by the paper found here:https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ <br>
I will reimplement U-Net in TF 2.0 which will be easier than others due to its use of TF and Keras already, it's just going to be a matter of using the same functions in TF 2.0. Looking at the repository, no pretrained weights were used in the calling of the unet function. Some pretrained weights were used for the github, but this model isn't supplied in the github repo, as such will likely train from scratch. The model was chosen due to it being an old model that won the 2015 ISIBI Challenge, it's also one of the most common examples of Semantic Segmentation.

#### Results of U-Net
The U-Net model is old now and this especially shows compared to some newer algorithms which are for the most part more efficient in terms of processing. U-Net with 33M+ parameters is hard to train even on V100s, its memory use didn't allow me to train at full resolution on the cityscapes dataset, thus I tried to modify U-Net to make it usable at full resolution. To do this I used Separable Convolutions instead of standard convolutions which can use a 10th of the number of parameters at times when compared to standard convolution. Also, made the number of filters/channels in each layer lower, making it a lightweight Separable U-Net. This produced a model of just over 300,000 parameters. However, despite these changes it still couldn't train at the full resolution. This little experiment of mine also failed at a lower resolution of 512x1024, its accuracy and IoU are very low. 

Training this model also took 2 minutes per epoch even with 8 V100's at the lower resolution. I went upto 125 epochs before I had seen enough to know this model wasn't going to be good.

My inituition is that, the model needs to be deeper and that there is a general limitation of how the U-Net is structured in the first place. With the model being deeper it would likely start to capture some of the smaller objects and finer objects like bicycles rather than just the large objects like the sky, vegetation and vehicles. The results of the best epoch (108) are below at 512x1024:
![sep_unet](https://github.com/SkyWa7ch3r/ImageSegmentation/blob/master/results/separable_unet/epoch_108.png?raw=true)

<b><u>IoU Over Classes on Validation Set</b></u>

classes       |  IoU   |  nIoU
--------------|--------|--------
road          | 0.910  |   nan
sidewalk      | 0.476  |   nan
building      | 0.740  |   nan
wall          | 0.000  |   nan
fence         | 0.000  |   nan
pole          | 0.000  |   nan
traffic light | 0.000  |   nan
traffic sign  | 0.000  |   nan
vegetation    | 0.845  |   nan
terrain       | 0.000  |   nan
sky           | 0.798  |   nan
person        | 0.006  | 0.006
rider         | 0.000  | 0.000
car           | 0.647  | 0.592
truck         | 0.000  | 0.000
bus           | 0.000  | 0.000
train         | 0.000  | 0.000
motorcycle    | 0.000  | 0.000
bicycle       | 0.000  | 0.000
<b>Score Average | <b>0.233  | <b>0.075

<b><u>IoU Over Categories</b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.944  |   nan
construction  | 0.750  |   nan
object        | 0.000  |   nan
nature        | 0.828  |   nan
sky           | 0.798  |   nan
human         | 0.006  | 0.006
vehicle       | 0.668  | 0.602
<b>Score Average | <b>0.571  | <b>0.304

### Bayesian SegNet
Following this github repo here: https://github.com/toimcio/SegNet-tensorflow <br>
Following this github repo here: https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/pooling.py
Which was inspired by the paper found here: https://arxiv.org/abs/1511.02680 <br>
Reimplementing this in TF2.0 should be ok, the author has done it in pure TF, redoing it in Keras will likely make it easier to understand. This model was chosen due to its use of VGG16 trained on ImageNet, this model was also the last hurrah of the standard Encoder-Decoder architecture which essentially mirrored the VGG16 model topping benchmarks in 2017. Trying this on the Cityscapes dataset and comparing it to newer models should show har far we have come.

#### Results of Bayesian SegNet
No current results, had trouble using the Max Pooling and Unpooling in keras, there are implementations out there, but I encountered many difficulties in this project. Thus, I ran out of time to fix this, perhaps some other time. Although, I suspect, the same issues as U-Net will arise here, the model will likely not run with High Resolution images well due to VGG16's size and computation power needed to train it. 

### DeepLabV3+
Following this github repo here: https://github.com/rishizek/tensorflow-deeplab-v3-plus <br>
Following this github repo here: https://github.com/bonlime/keras-deeplab-v3-plus <br>
Following this github repo here: https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0 < br>
Which was inspired by this paper found here: https://arxiv.org/abs/1802.02611 <br>
One of the many models which used Atrous Spatial Pyramid Pooling (ASPP), it beat the previous DeepLabV3 and PSPNet which employed very similar architectures for semantic segmentation. It actually uses a modified Xception as its backbone, with an encoder featuring the ASPP with a novel decoder. It was chosen as it seems like a good representation of this type of model. Using this on Cityscapes will be more like reporducing the results from the original paper. Overall this will add to the validity of this benchmark.

#### Results of DeepLabV3+
The Results of DeeplabV3+ were interesting, first, the model is too large with its Modified Aligned Xception backbone to train at the full resolution of 1024x2048 in my experience using TensorFlow 2.1 and Keras. The authors of DeepLabV3+ used a resolution of 513x513 for their training getting ~79% on the validation set. Compared to models from 2019, its clear that models from 2018 and below are too large and not efficient enough to run at full resolution. 

With that said, I decided to try and modify DeepLabV3+ to run at full resolution. To do this, I made sure separable convolutions were used in the decoder to reduce parameters and I used MobileNetV2 with ImageNet weights as the backbone. This has around 4.4M+ parameters which is a 10th of the original Modified Aligned Xception DeepLabV3+. I trained the MobileNetV2 Backbone model for 500 epochs at 2 minutes an epoch with 8 V100's for ~17 hours.

This model produced the results at epoch 490 below:
![deeplab](https://github.com/SkyWa7ch3r/ImageSegmentation/blob/master/results/deeplabv3+/epoch_490.png?raw=true)

<b><u>IoU Over Classes on Validation Set</b></u>

classes | IoU | nIoU
--------|-----|-----
road    | 0.885 | nan
sidewalk| 0.523 | nan
building| 0.815 | nan
wall    | 0.104 | nan
fence   | 0.301 | nan
pole          | 0.281 |   nan
traffic light | 0.295 |   nan
traffic sign  | 0.426 |   nan
vegetation    | 0.860 |   nan
terrain       | 0.427 |   nan
sky           | 0.837 |   nan
person        | 0.547 | 0.306
rider         | 0.090 | 0.037
car           | 0.798 | 0.669
truck         | 0.026 | 0.009
bus           | 0.085 | 0.036
train         | 0.106 | 0.045
motorcycle    | 0.088 | 0.033
bicycle       | 0.551 | 0.336
<b>Score Average | <b>0.423 | <b>0.184

<b><u>IoU Over Categories </b></u>

categories    |   IoU   |   nIoU
--------------|---------|--------
flat          | 0.927   |   nan
construction  | 0.824   |   nan
object        | 0.356   |   nan
nature        | 0.863   |   nan
sky           | 0.837   |   nan
human         | 0.574   | 0.327
vehicle       | 0.789   | 0.664
<b>Score Average |<b> 0.739 | <b>0.495


### Fast-SCNN
Following this github repo here: https://github.com/kshitizrimal/Fast-SCNN <br>
Which was inspired by this paper found here: https://arxiv.org/abs/1902.04502 <br>
First this one is already using TF 2.0 and Keras, so my job here is making the Cityscapes work with it, which shouldn't be an issue. It's one of the most recent semantic segmentation articles being from 2019. It was designed to has very fast inference and training times. In fact it can run at 120+ fps when using 1024x2048 images which is *impressive*. Also, it was designed in mind that embedded systems would use this model for inference. It's done this using the power of Depthwise Convolution and very efficient pyramid modules. While its accuracy isn't as high as others, it's able to achieve 68% on Cityscapes while infering in *real time*. Ultimately will be interesting to test against others when not in real time as well.

#### Results of Fast-SCNN
The Fast-SCNN out of the ones tested here, is by far the best at full resolution, it would be beaten at lower resolutions but its training would be much faster than the others. I was able to train this model at full resolution with a batch size of 4 per GPU. Using 8 V100's an epoch only took 1 minute, meaning this model trained in ~8 hours which is much faster than the other models I trained. In this case I was able to get close to replicate the author's scores here. Had I trained it for longer, I believe I would've replicated the authors results, many thanks to the original author of the code kshitizrimal, I modified it a little to handle lower resolutions without anyone using having to modify the model in code. 

The best result was at epoch 454 which is below:
![fastscnn](https://github.com/SkyWa7ch3r/ImageSegmentation/blob/master/results/fastscnn/epoch_454.png?raw=true)

<b><u>IoU Over Classes on Validation Set</b></u>

classes       |  IoU  |   nIoU
--------------|-------|---------
road          | 0.943 |    nan
sidewalk      | 0.761 |    nan
building      | 0.876 |    nan
wall          | 0.444 |    nan
fence         | 0.433 |    nan
pole          | 0.414 |    nan
traffic light | 0.511 |    nan
traffic sign  | 0.595 |    nan
vegetation    | 0.889 |    nan
terrain       | 0.546 |    nan
sky           | 0.908 |    nan
person        | 0.667 |  0.396
rider         | 0.437 |  0.228
car           | 0.899 |  0.787
truck         | 0.552 |  0.192
bus           | 0.650 |  0.365
train         | 0.451 |  0.187
motorcycle    | 0.380 |  0.176
bicycle       | 0.631 |  0.351
<b>Score Average | <b>0.631 | <b>0.335

<b><u>IoU Over Categories </b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.948  |   nan
construction  | 0.882  |   nan
object        | 0.494  |   nan
nature        | 0.893  |   nan
sky           | 0.908  |   nan
human         | 0.678  | 0.418
vehicle       | 0.878  | 0.748
<b>Score Average | <b>0.812  | <b>0.583


### Fast-FCN
Following this github repo here: https://github.com/wuhuikai/FastFCN <br>
Which was inspired by this paper found here: http://wuhuikai.me/FastFCNProject/fast_fcn.pdf
This architecture is similar to DeepLabV3+ however it uses Joint Pyramid Upsampling (JPU) for its decoding of images which is more efficient in its upsampling compared to atrous convolution. It also showed that using the JPU had no real performance loss alongside other reductions in computational complexity make it another great contender for using on embedded systems like the Jetson AGX. The ResNet-50 and ResNet-101 are used as the backbones here, meaning the weights will start from a pretrained ImageNet ResNet as most do in 2019. 

## Conclusions So Far
As of the start of 2020, many models are having a lot of trouble running at full resolution of 1024x2048, the only successful one I've tested so far is Fast-SCNN. Although, I suspect Fast-FCN will give it a run for its money. Many models before 2019 are simply to large and inefficient with their layers to produce good results at a high resolution. Using high resolution will only become more important as autonomous vehicles will likely want to use high resolution cameras as their eyes on the road. If we want to use Semantic Segmentation, or the even more computationally expensive process of combining Semantic and Instance Segmentation aka Panoptic Segmentation then our models will need to be a lot more efficient. The Fast-SCNN model I have running on the NVIDIA Jetson Xavier, which is performing terribly likely due to how I've converted Fast-SCNN to ONNX is only running at 10FPS, but I hope this will improve as I continue to work on it (hopefully).

## Personal Comments on TensorFlow at the start of 2020 (TF2.1)
After so many issues I had during this experiment, I felt the need to create this section. The documentation needs to be updated to reflect the major changes TensorFlow has made during the move from version 1. Also, MirroredStrategy is slower than Horovod for distributed training, so more work needs to be done there. Also, if anyone at Google is reading this, MultiWorkerMirroredStrategy doesn't work in a container if you need to use SLURM to submit a job on a machine. Horovod on the other hand, works very well in this environment, its easy to see why many use it even when running Multi-GPU on a single machine instead of across nodes.

Also, please fix mixed precision with eager training, this would have likely sped up training and allowed higher batch sizes, but instead made it slower while allowing a larger batch size. 

Overall, I like the direction of TensorFlow as it is today but a lot more work has to be done to get the best performance I can get with the models I'm using here.

## TODO

- [ ] Fix and Train Bayesian SegNet at 1024x1024 if possible
- [ ] Implement Fast-FCN in TensorFlow 2.1 at full resolution
- [ ] Improve Separable Unet with more layers and more filters/channels per layer
- [ ] Implement and Create a README for HRNetV2 + OCR + SegFix the current leader on the Cityscapes Dataset
- [ ] Train the Modiefied Aligned Xception on ImageNet, and freeze some of the layers in an effort to train at full resolution
- [ ] Create ONNX files for NVIDIA Jetson AGX Xavier
- [ ] Use ONNX files to produce predictions from images taken in Perth, Western Australia.
- [ ] Fine Tune the trained models with the Coarse Cityscapes Dataset
- [ ] Implement a dataset for the Coarse Images

## Datasets
* Cityscapes - will require an account. https://www.cityscapes-dataset.com/downloads/

## Acknowledgements
For all the github repos I have referenced, thank you, you helped make this project possible. I'd also like to thank the Pawsey Supercomputing Centre for access to their new Topaz Cluster with Tesla V100's available, this made the training feasible. Also many thanks to my supervisors at Curtin University for their support and the opportunity to work on this project.
