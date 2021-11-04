### QQ浏览器视频多模态相似度比赛
这是QQ浏览器视频多模态相似度比赛的官方tensorflow版本baseline代码

#### 1. 数据
请从腾讯微云下载data目录，数据格式说明见赛题描述

#### 2. 代码基础结构
- 所有参数配置在 [config.py](config.py)
- 训练入口在 [train.py](train.py)
- baseline模型结构见 [model.py](model.py)
- 数据预处理和解析见[data_helper.py](data_helper.py)
- 训练过程的指标输出见[metrics.py](metrics.py)
- 生成提交文件的代码参看[inference.py](inference.py)
- 评测代码参看[evaluate.py](evaluate.py)

#### 3. Train
```bash
pip install -r requirements.txt
python train.py
```

#### 4. Inference
```bash
python inference.py
```

#### 5. Evaluate
```bash
python evaluate.py
```
