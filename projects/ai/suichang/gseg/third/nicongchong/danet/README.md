# DANet-keras
**原文地址：[Dual Attention Network for Scene Segmentation](<https://arxiv.org/abs/1809.02983>)**

**[论文阅读笔记](<https://niecongchong.github.io/2019/08/05/DANet-%E5%8F%8C%E9%87%8D%E6%B3%A8%E6%84%8F%E5%8A%9B%E8%9E%8D%E5%90%88%E7%BD%91%E7%BB%9C/>)**

**源代码pytorch：https://github.com/junfu1115/DANet/**

### 代码组织形式

* `train.ipynb:`模型训练，包含超参设置、模型调用、训练、可视化。
* `test_crop_image.py:`模型测试，包含模型加载、测试、可视化。
* `dataloaders/generater.py:`数据加载，数据路径获取、图片读取、预处理及在线扩充。
* `model/danet_resnet101:`模型定义。
* `layers/attention:`PAM空间注意力和CAM通道注意力模块搭建。
* `utils/loss.py:`损失函数，包含`dice_loss、ce_dice_loss、jaccard_loss(IoU loss)、ce_jaccard_loss、tversky_loss、focal_loss`
* `utils/metrics.py:`评价指标，包含`precision、recall、accuracy、iou、f1`等。
* `train.html:`训练过程记录，保存为html文件。
