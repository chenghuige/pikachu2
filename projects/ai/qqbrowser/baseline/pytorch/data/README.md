#### data目录说明

- chinese_L-12_H-768_A-12 存放bert pytorch模型，具体包括预训练模型 bert_google.bin 和模型训练使用词表 vocab.txt 
    - baseline使用bert base预训练模型，参赛者可以根据需求替换或变更。例如使用[模型和bert词表](https://huggingface.co/bert-base-chinese/tree/main)
    - 使用其它版本bert模型需要在../embedding_example.yaml中匹配配置。具体涉及REPRESENTATION 和 DATASET config
     
- tag_list.txt 是用于多标签分类的tag子集，用于构建示例训练目标。参赛者可以根据预训练数据，自由构建预训练模型目标。具体构造可以使用tag_id,category_id等，比赛对此不做限定。
- pointwise 是100w预训练数据集数据
- pairwise 验证集数据
- test_a 是初赛的测试集
- test_b 是复赛的测试集，在复赛阶段 release

#### 具体目录结构
├── tag_list.txt  
├── desc.json  
├── chinese_L-12_H-768_A-12  
│   ├── vocab.txt  
│   ├── bert_google.bin   
├── pairwise  
│   ├── label.tsv  
│   └── pairwise.tfrecords  
├── pointwise  
│   ├── pretrain_0.tfrecords  
│   ├── pretrain_10.tfrecords  
│   ├── pretrain_11.tfrecords  
│   ├── pretrain_12.tfrecords  
│   ├── pretrain_13.tfrecords  
│   ├── pretrain_14.tfrecords  
│   ├── pretrain_15.tfrecords  
│   ├── pretrain_16.tfrecords  
│   ├── pretrain_17.tfrecords  
│   ├── pretrain_18.tfrecords  
│   ├── pretrain_19.tfrecords  
│   ├── pretrain_1.tfrecords  
│   ├── pretrain_2.tfrecords  
│   ├── pretrain_3.tfrecords  
│   ├── pretrain_4.tfrecords  
│   ├── pretrain_5.tfrecords  
│   ├── pretrain_6.tfrecords  
│   ├── pretrain_7.tfrecords  
│   ├── pretrain_8.tfrecords  
│   └── pretrain_9.tfrecords  
├── test_a  
│   └── test_a.tfrecords   
└── test_b  
    └── test_b.tfrecords  