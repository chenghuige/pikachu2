# HRNet-keras-semantic-segmentation

**HRNet v1：[Deep High-Resolution Representation Learning for Human Pose Estimation](<https://arxiv.org/abs/1902.09212>)**

**HRNet v2：[High-Resolution Representations for Labeling Pixels and Regions](<https://arxiv.org/abs/1904.04514>)**

**[HRNet v1 论文阅读笔记](<https://niecongchong.github.io/2019/05/13/HRNet%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0%E5%8F%8A%E4%BB%A3%E7%A0%81%E7%90%86%E8%A7%A3/>)**

**[HRNet v2 论文阅读笔记](<https://niecongchong.github.io/2019/07/01/HRNetv2%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/>)**

**HRNet v1 源代码 pytorch：https://github.com/leoxiaobin/deep-high-resolution-net.pytorch**

**HRNet v2 源代码 pytorch：https://github.com/HRNet/HRNet-Semantic-Segmentation**

### 代码组织形式

* `train.ipynb:`模型训练，包含超参设置、模型调用、训练、可视化。
* `test_crop_image.py:`模型测试，包含模型加载、测试、可视化。
* `dataloaders/generater.py:`数据加载，数据路径获取、图片读取、预处理及在线扩充。
* `model/seg_hrnet:`模型定义。
* `utils/loss.py:`损失函数，包含`dice_loss、ce_dice_loss、jaccard_loss(IoU loss)、ce_jaccard_loss、tversky_loss、focal_loss`
* `utils/metrics.py:`评价指标，包含`precision、recall、accuracy、iou、f1`等。
* `train.html:`训练过程记录，保存为html文件。
