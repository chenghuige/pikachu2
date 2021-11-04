# https://algo.browser.qq.com/ 第三名解决方案    
[文档](https://note.youdao.com/s/WlmA0aUJ) <br>   
赛后借鉴gdy的label归一化和mlm方案，以及[第一名解决方案](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st) 做了进一步迭代。  
其中label归一化相对比赛中采用的归一化效果更好，能继续提升2-3k。    
融合vision和title的mlm相对效果更好，继续提升1-2k，默认的预训练已经改为--use_vision  
尝试采用第一名解决方案的多任务方式，目前支持tag+mlm，暂未实现mask frame。   
相对文档时候的单模型离线0.8298，目前单模型离线0.84。 
由于采用了较为严格的验证方案（严格保证训练和验证没有重复的vid,可以对比第一名的验证方案并没有采用严格的去除重复vid会造成离线分数过高)，因此预期线上结果比较一致，待后续官方数据集开放可做进一步验证。    
另外基于word/char混合方案实验中。    

最快单模型流程：  
## 数据预处理  
cd pikachu/projects/ai/qqbrowser/tools  
python dump-ids.py  
python dump-infos.py  
cd ../prepare  
python gen-vocabs.py  
cd ../jupyter  
python display.py 
python pairwise-label.py  
python w2v.py  
cd ../prepare  
cd ai/projects/qqbrowser/prepare  
## 生成 ../input/tfrecords0  约60分钟  
sh gen-records-v0.sh 0  
cd ../pretrain   
## mlm 预训练 4卡A100 约80分钟  
sh run/0/chinese-roberta-wwm-ext.sh    
cd ../qqbrowser   
## pointwise（约150分钟） + pairwise (单个fold约10分钟，总共5folds约60分钟)  
//注意默认设置--gpus=2可以设置 --gpus=-1使用所有gpu 或者指定其他数目  

//pointwise只采用tag  约0.832 - 0.834  
./offline.sh run/40/model2.sh  

//简化fusion 最后只用bert输出过nextvlad接fc 待验证  
./offline.sh run/40/model2.sh  --layer_norm=0 --activation='' --incl=merge

//large版本 待验证   
cd ../pretrain   
sh run/0/chinese-roberta-wwm-ext-large.sh     
cd ../qqbrowser  
./offline.sh run/40/model2.sh --hug=large  

//该方案pointwise为tag+mlm 待验证  
./offline.sh run/40/model2.mlm.sh    

//word版本 待验证  
cd ai/projects/qqbrowser/prepare   
//注意设置 FLAGS.mix_segment=True  
sh gen-records-v0.sh 0   
cd ../pretrain   
sh run-word/0/base.sh       
./offline.sh run/40/model2.word.sh    
