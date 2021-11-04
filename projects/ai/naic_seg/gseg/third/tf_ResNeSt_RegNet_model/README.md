## Introduction
 Currently support tensorflow in 
 - **ResNeSt**  2d&3d
 - **RegNet**
 - **DETR** (modified classfication)
 - **GENet** (2020 GPU-Efficient Network)
 
model only, no pertrain model for download (simply not enough free time and resource).  
easy to read and modified. welcome for using it, ask question, test it, find some bugs maybe.

ResNeSt based on [offical github](https://github.com/zhanghang1989/ResNeSt) .

## Update
**2020-7-30**: Add and based on [GENet: A GPU-Efficient Network](https://github.com/idstcv/GPU-Efficient-Networks). Paper shows achieve good performence under GPU environment, very similar to RegNet. model name `GENet_light`, `GENet_normal`, `GENet_large`. 

**2020-6-14**: Add **Resnest3D**, thanks to @vitanuan, model name `resnest50_3d`, `resnest101_3d`, `resnest200_3d`, input shape is 4d like `input_shape = [50,224,224,3]`

**2020-6-5**: Add **DETR** (res34, resNest50 backbone) **End-to-End Object Detection with Transformers**, Experiment and inovation model, i slightly modified it into a classficaiton verison. Free to try.

**2020-5-27**: ResNeSt add [CB-Net](https://arxiv.org/pdf/1909.03625.pdf) style to enahce backbone. theoretically, it should improve the results. Wait for test.

## Usage
usage is simple:
```
from models.model_factory import get_model

model_name = 'ResNest50'
input_shape = [224,244,3]
n_classes = 81
fc_activation = 'softmax'
active = 'relu' # relu or mish

model = get_model(model_name=model_name,
                  input_shape=input_shape,
                  n_classes=n_classes,
                  fc_activation=fc_activation,
                  active=active',
                  verbose=False,
                 )

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
model.fit(...)
```


- if you add CB_Net in ResNeSt: add `using_cb=True` like:
```
"""
Beware that if using CB_Net, the input height and width should be divisibility of 
2 at least 5 times, like input [224,448]: 
[224,448]<->[112,224]<->[56,112]<->[28,56]<->[14,28]<->[7,14]
correct way:
[224,224] downsample->[112,112],
[112,112] upsample->[224,224],
then [224,224]+[224,224]

incorrect way:
[223,223] downsample->[112,112],
[112,112] upsample->[224,224],
[223,223] != [224,224] cannt add
"""

model = get_model(...,using_cb=True)
```
- DETR experiment model, free to modified the transformer setting.
```
# ResNest50+CB+transfomer looks powerful! but heavily cost.
model_name = 'ResNest50_DETR' 

#res34 not implement using_cb yet, it supporse to be a lighter verison.
model_name = 'res34_DETR' 

model = get_model(...,
                  hidden_dim=512,
                  nheads=8,
                  num_encoder_layers=6,
                  num_decoder_layers=6,
                  n_query_pos=100)
```


## Models 
models now support:
```
ResNest50
ResNest101
ResNest200
ResNest269

RegNetX400
RegNetX1.6
RegNetY400
RegNetY1.6
AnyOther RegNetX/Y

res34_DETR
ResNest50_DETR
```
#### RegNet
for RegNet, cause there are various version, you can easily set it by `stage_depth`,`stage_width`,`stage_G`.

```
#RegNetY600
model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
                verbose=True,fc_activation=fc_activation,stage_depth=[1,3,7,4],
                stage_width=[48,112,256,608],stage_G=16,SEstyle_atten="SE")

#RegNetX600
model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
                verbose=True,fc_activation=fc_activation,stage_depth=[1,3,5,7],
                stage_width=[48,96,240,528],stage_G=24,SEstyle_atten="noSE")
```

details seting (from orginal paper ):
- [facebookresearch/pycls](https://github.com/facebookresearch/pycls)

![alt text](https://raw.githubusercontent.com/QiaoranC/tf_ResNeSt_RegNet_model/master/readme_img/regnet_setting.png)


- CB-Net, using this style to enhace ResNest
![alt text](https://raw.githubusercontent.com/QiaoranC/tf_ResNeSt_RegNet_model/master/readme_img/CBNet.png)

#### DETR
-  orginal is detection version with two linear out (box and cls), here only clsficaiton (with some modification), if anyone interesting in the detection version, asked and i can change to a deteciton version. 
 - torch version using `nn.Parameter` to buildt `q,k,v`, but in tensorflow2.x, i tried similar `Variable`, but the `Variable` shape can't set batch dimed like `(?,100,100)`, and model layers output are like (?,7,7,2048), So to combine Variable into model, i use a trick way, to make a fake layer out `(?,100,100)`, and add with the `Variable` (100,100),  Variable will become `(?,100,100)` then can feed into `transformer`. I didnot found a better way. Again, Welcome any good ideal or suggesions.

## Discussion
I compared **ResNeSt50** and some **RegNet**(below 4.0GF) in my own project, also compared to **EfficientNet b0/b1/b2**.
it seems **EfficientNet** is still good at balance in size/speed and accuracy, and **ResNeSt50** performe well at accuarcy also lower in size/speed, And **RegNet** not that fast and acuracy not that good, seems normal.
