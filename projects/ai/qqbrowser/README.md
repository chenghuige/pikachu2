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

本方案和baseline基本保持一致：  
模型 | Pointwise | Pairwise(5Fold平均) | 差值  
---|--- | --- | ---
赛中离线最佳单模型(roberta.rv1.400) | 0.7284 |  0.8298 | 0 
改为bin norm的label | _ |  0.832 | +2k
mlm的预训练加入视频(模型结构相应也加入视频融合) | _ |  0.834 | +2k
去掉layernorm | 0.6605 | 0.656 | -173k
pointwise多分类(softmax loss)->多标签(sigmoid loss) | 0.7175 | 0.7827 | -47k
去掉NextVlad Merge部分 |0.7283 | 0.823 | -6.8k
多Linear平均(multi-drop with drop rate 0) -> 单Linear | 0.7354 | 0.8243 | -5.5k
不使用mlm cotinue pretrain| 0.7288 | 0.8261 | -3.7k
去掉Words输入部分 |0.7269 | 0.8262 | -3.6k
dice -> relu | 0.7318| 0.8262 | -3.6k
添加SE | 0.7283 | 0.8262 | -3.6k
NextVlad groups 4 -> 8 | 0.7287 | 0.8264 | -3.4k 
去掉label和样本权重策略| 0.7295 | 0.8276 | -2.2k

个人觉得最满意的修改是 tag预测部分，做了较多实验，下面的方案相对baseline提升应该不止47K。  
上面的消融只是说softmax和sigmoid区别， 而我还做了从部分tag到全tag使用的修改在初期验证也是有提升的具体数值没有记录。   
具体方案，所有tag都参与训练，每个tag id 对应一个向量。  
对应video的最终向量。目标是使得video向量和当前video标注的各个tag id向量尽可能接近而和其他tag向量尽可能远离。    
具体做法比如一个video A， 有三个tag标注 [pos1,pos2,pos3] 因为每个video的tag数目不固定，我们padding 0到定长 M， 通过Log Neg sampling 我们sample得到个N负例tag id。    
因此我们得到[pos1,pos2,pos3,0,0，....0] 长度为M 正例tag  [neg1,neg2,neg3, ..... negN]长度为N 负例tag    
假定video A的最终向量为emb   
那么我们对应pos1,pos2,pos3计算三次softmax loss取平均（因为实际padding到了M可以利用mask mask掉padding 0的部分）,以 pos1 为例（其他类似）  
label为 [1,0,0....,0] N个0     
pred为 [dot(pos_tag1,emb),dot(neg_tag1,emb),dot(neg_tag2,emb)...,dot(neg_tagN,emb)]    


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

