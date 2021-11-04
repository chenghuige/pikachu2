# WeChat_Big_Data_Challenge_DeepCTR_baseline

比赛官网：[2021中国高校计算机大赛 微信大数据挑战赛](https://algo.weixin.qq.com/) 

## 【更新】关于复现不出0.63的问题
群里有很多小伙伴提出无法复现0.63，跑出来结果只有0.5多。下面是我这边具体的环境，麻烦大家再试下（应该是tf版本不同导致的）：
```
deepctr==0.8.5
numpy==1.16.4
pandas==0.24.2
tensorflow==1.12.0
scikit_learn==0.24.2
```


目前来看 **tf 1.12.0或1.13.1**是可以正常复现出0.63的。以下是一些同学的反馈：
![ ](https://mmbiz.qpic.cn/mmbiz_png/B5C8lMMBfXtic8UBJiahA72wG65kvywVpeZXJ0aFskA5nzMN4XibiaLmear9ERAgGmX9SUdonzwGDT7RwUqvldW6yg/0?wx_fmt=png)
![ ](https://mmbiz.qpic.cn/mmbiz_png/B5C8lMMBfXtic8UBJiahA72wG65kvywVpeTIic3fV869SbFzFfqt1FxgSIeFjUP9x408Ij5RyEQJw2xrJXsoJq6zw/0?wx_fmt=png)
![ ](https://mmbiz.qpic.cn/mmbiz_png/B5C8lMMBfXtic8UBJiahA72wG65kvywVpeib2dGAWKM0mFSAgeZibAt6D3VRUvq0lslUokMEqkkiaLpibiczb55uamsLg/0?wx_fmt=png)

## 方案说明

本方案为基于[DeepCTR](https://github.com/shenweichen/DeepCTR)实现的多任务学习模型[MMOE](https://dl.acm.org/doi/10.1145/3219819.3220007)

- 特征：本方案关注于模型，仅使用以下6个较为基础的原始特征：['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds']


- 线上结果：

  | 得分     | 查看评论 | 点赞     | 点击头像 | 转发     |
  | -------- | -------- | -------- | -------- | -------- |
  | 0.633475 | 0.612465 | 0.613346 | 0.700479 | 0.643898 |


## 运行环境
 python 3.6
 deepctr==0.8.5
 tensorflow-gpu(tensorflow)
 pandas
 scikit-learn

### deepctr安装说明
- CPU版本
  ```bash
  $ pip install deepctr==0.8.5
  ```
- GPU版本
  先确保已经在本地安装`tensorflow-gpu`,版本为 **`tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*`**，然后运行命令
    ```bash
    $ pip install deepctr==0.8.5 --no-deps
    ```

## 运行说明
1. 新建data目录，下载比赛数据集，放在data目录下并解压，得到wechat_algo_data1目录

2. ```bash
   python run_mmoe.py
   ```


## 运行时间

在Tesla P40 24G GPU、E5-2650 v4 CPU机器上，训练时间为205s/epoch（6708846条样本）。

训练时显存占用为695MiB，内存占用为2.2G。

预测时间如下：

```
4个目标行为421985条样本预测耗时（毫秒）：352.331
4个目标行为2000条样本平均预测耗时（毫秒）：1.670
```

## 关于DeepCTR

[DeepCTR](https://github.com/shenweichen/DeepCTR)是一个易用、可扩展的深度学习点击率预测算法包，基于tensorflow深度学习框架。

添加**特征**时，仅添加feature_columns即可，无需改动模型；

在**模型**方面，DeepCTR包含20多个CTR模型（如DeepFM、xDeepFM、DCN、AutoInt、DIN、FiBiNET等），可直接通过模型名调用。如需**自定义模型**，DeepCTR中也有很多高复用性的模块（例如DNN、FM、BiInteractionPooling、CIN、CrossNet等）。更多使用方法请参考[DeepCTR文档](http://deepctr-doc.readthedocs.io/)。

DeepCTR也有pytorch版本：DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch

（想用torch版的小伙伴在下方评论区扣1，人数多的话，后面实现一波torch版的MMOE）。

## 其他

楼主之前做比赛，从很多前辈的分享或开源中学到了很多，现在希望自己也能贡献一些。希望本文可以帮助大家降低深度学习CTR模型的门槛，普及这些算法技术。

预祝大家在本次大赛中取得好成绩！
