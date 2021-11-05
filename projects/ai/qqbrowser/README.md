# https://algo.browser.qq.com/ 第三名解决方案    
[文档](https://note.youdao.com/s/WlmA0aUJ) <br>   
赛后借鉴gdy的label归一化和mlm方案，以及[第一名解决方案](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st) 做了进一步迭代。  
其中label归一化相对比赛中采用的归一化效果更好，能继续提升2-3k。这个提升稳定的，相对如果不采用任何label处理提升约6k。   
融合vision和title的mlm相对效果更好，继续提升1-2k，默认的预训练已经改为--use_vision 。  
尝试采用第一名解决方案的多任务方式，目前支持tag+mlm，暂未实现mask frame,不过并没有获得提升。     
相对文档时候的单模型离线0.8298，目前单模型离线0.834。 
由于采用了较为严格的验证方案（严格保证训练和验证没有重复的vid,如果不去除重复vid会造成离线分数过高)，因此预期线上结果比较一致，待后续官方数据集开放可做进一步验证。    
值得注意的是，pointiwse和pairwise的spear结果有强相关性但不是完全对齐，这会给迭代带来困难，因为pointwise还相对比较耗时。   
Offline过程的pointwise没有使用pairwise数据，Online过程的pointwise是否使用pairwise数据对在线的影响没有提交验证过。 如果不使用的话离线在线只有pairwise区别简洁一些。  

最快单模型流程：  
## 数据预处理  
cd projects/ai/qqbrowser/tools  
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

//large版本 并没有验证提升  
cd ../pretrain   
sh run/0/chinese-roberta-wwm-ext-large.sh     
cd ../qqbrowser  
./offline.sh run/40/model2.sh --hug=large  

//该方案pointwise为tag+mlm 并没有验证提升..  
./offline.sh run/40/model2.mlm.sh    

