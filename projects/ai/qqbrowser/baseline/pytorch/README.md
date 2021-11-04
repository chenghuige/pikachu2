### QQ浏览器视频多模态相似度比赛
- 这是QQ浏览器视频多模态相似度比赛的官方lichee框架版本baseline代码
 
#### 数据
- 请从腾讯微云下载data目录，数据格式见赛题描述
 
#### 依赖
```bash
pip install -r requirements.txt
```

#### 代码目录结构
- 配置文件: [embedding_example.yaml](embedding_example.yaml)
- 主函数：[main.py](main.py)
- 示例代码依赖：[module](module)
- 数据文件夹：[data](data) ,数据结构见数据文件夹说明
- tfrecords python解析示例代码：[read_tf_record_example.py](read_tf_record_example.py)
  
#### 训练&测试：
- 训练：
```bash
python main.py --trainer=embedding_trainer --model_config_file=embedding_example.yaml
```
- 测试需要指定checkpoint和指定配置文件，示例如下：
```bash
python main.py --trainer=embedding_trainer --model_config_file=your_config.yaml --mode eval --checkpoint your_check_point.bin --dataset SPEARMAN_DATA
```

