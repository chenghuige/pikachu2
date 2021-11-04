# CCF BCDI2017 卫星影像的AI分类与识别 线上Top1部分代码

# 运行环境
python版本：python2.7

依赖库：pytorch,torchvision,[visdom](http://github.com/facebookresearch/visdom)，[pydensecrf](http://github.com/lucasb-eyer/pydensecrf)...

# 0.Overview
* DenseNet121为基础网络，PSPNet作为分割的模型，多尺度训练/测试，CRF后处理等
* 训练数据集：

	* 初赛训练数据s1
	* 复赛训练数据s2
	* 初赛训练数据+复赛训练数据 s1s2
	* 初赛训练数据+CRF处理复赛训练数据（是s1s2-crf2）

* 主要尝试的模型:

	* 训练数据集s1s2 (pspnet-densenet-s1s2)
	* 训练数据集s1s2-crf2 (pspnet-densenet-s1s2-crf2)
	* 不同网络输入尺度 (pspnet-densenet-s1s2-320)
	* focal loss (pspnet-densenet-s1s2-crf2-fl)
	* 类别加权训练 (pspnet-densenet-s1s2-crf2-weight)

# 1.数据预处理
* [数据下载](https://pan.baidu.com/s/1nu8srUh) password：al0x
* 将训练数据放入工程目录下dataset/CCF-training和dataset/CCF-traing-Semi下
测试数据dataset/CCF-testing-Semi, 修改utils/preprocess.py中的工程目录 `ProjectDir`
* 执行 ./preprocess.sh (可能时间比较久...）

# 2.训练
run_train.sh 根据Overview里面的模型设置，更改train_dir选择对应的训练数据和model_name设置训练的模型
* pspnet-densenet-s1s2-320,更改--image_rows 和 --img_cols 为320
* 在run_train.sh，除了pspnet-densenet-s1s2-crf2-fl调用 train-fl.py, 其它模型用train.py
* 对于pspnet-densenet-s1s2-crf2-weight,更改train.py中的weights_per_class 为[0,1,1,3,3]，默认[0,1,1,1,1]

# 3.测试 & Vote

* run_test.sh  更改model_name选择对应的模型测试
* run_vote.sh  更改model_name,对同一模型的不同epoch测试结果进行投票，得到该单模型结果
* ./submit.sh  每个模型的测试目录 use_crf（e.g. ./submit.sh results/pspnet-densenet-s1s2-crf2/vote 1）
