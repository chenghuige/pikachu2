#model_dirs = [
# 'sm.Unet.EfficientNetB5.augl4.size288.mrate1.300epoch.largefilters.1120',
# 'sm.Unet.EfficientNetB4.augl4.size288.mrate1.300epoch.1117',
# 'sm.Unet.EfficientNetB2.augl4.size352.mrate1.300epoch.1122',
# 'sm.Unet.EfficientNetB2.augl4.size320.mrate1.300epoch.largefilter.1122',
# 'sm.FPN.EfficientNetB4.augl4.size256.mrate1.300epoch.1117',
# 'sm.Linknet.EfficientNetB4.augl4.size288.mrate1.300epoch.largefilter.1123'
#]

# 训练单模型 共6个 可能3-4个模型能够确保前10
#rm -rf ../working/quarter
#bin=./train/quarter/train.sh 
#sh $bin --mn=sm.Unet.EfficientNetB5.augl4.size288.mrate1.300epoch.largefilters.1120 
#sh $bin --mn=sm.Unet.EfficientNetB4.augl4.size288.mrate1.300epoch.1117 
#sh $bin --mn=sm.Unet.EfficientNetB2.augl4.size352.mrate1.300epoch.1122
#sh $bin --mn=sm.Unet.EfficientNetB2.augl4.size320.mrate1.300epoch.largefilter.1122
#sh $bin --mn=sm.FPN.EfficientNetB4.augl4.size256.mrate1.300epoch.1117 
#sh $bin --mn=sm.Linknet.EfficientNetB4.augl4.size288.mrate1.300epoch.largefilter.1123 

# 模型转换 生成 model_lr.h5.. model_rot3.h5 一共6个待选model 用于选择集成
#rm -rf ../working/convert
#FAST=1 python ./tools/converts.py ../working/quarter 

# 模型集成 将选择好的6个单模型 组合到一个model.h5模型
rm -rf ../working/ensemble
python ./tools/ensemble.py 

