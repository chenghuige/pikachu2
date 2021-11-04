## 数据处理
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
#tfrecords0  
sh gen-records-v0.sh 0   
#tfrecords1     
sh gen-records-v1.sh 1    

## 预训练
### 基于char的bert模型mlm继续训练
cd projects/ai/qqbrowser/pretrain  
sh run/0/chinese-macbert-base.sh
sh run/0/chinese-roberta-wwm-ext.sh  
sh run/0/bert-base-chinese.sh  

## 训练(offline)
cd projects/ai/qqbrowser/qqbrowser    
#如果只复现test结果可以不跑offline的pairwise部分   
sh ./offline.sh run/40/model.sh --hug macbert --wes 256 --rv 0  
sh ./offline.sh run/40/model.sh --hug macbert --wes 256 --rv 1  
sh ./offline.sh run/40/model.sh --hug macbert --wes 400 --rv 0  
sh ./offline.sh run/40/model.sh --hug macbert --wes 400 --rv 1  
sh ./offline.sh run/40/model.sh --hug roberta --wes 256 --rv 0  
sh ./offline.sh run/40/model.sh --hug roberta --wes 256 --rv 1  
sh ./offline.sh run/40/model.sh --hug roberta --wes 400 --rv 0  

#rv可选 0，1  
#wes可选 256，400  
#hug可选 macbert,roberta,base  
#最佳单模型参数 --rv=1 --wes=400 --hug=roberta

## 训练(online)和集成  
### 训练(online)  
 sh ./online.sh run/40/model.sh --hug macbert --wes 256 --rv 0  
 sh ./online.sh run/40/model.sh --hug macbert --wes 256 --rv 1  
 sh ./online.sh run/40/model.sh --hug macbert --wes 400 --rv 0  
 sh ./online.sh run/40/model.sh --hug macbert --wes 400 --rv 1  
 sh ./online.sh run/40/model.sh --hug roberta --wes 256 --rv 0  
 sh ./online.sh run/40/model.sh --hug roberta --wes 256 --rv 1  
 sh ./online.sh run/40/model.sh --hug roberta --wes 400 --rv 0  
 #最佳单模型  
 sh ./online.sh run/40/model.sh --hug roberta --wes 400 --rv 1  
 sh ./online.sh run/40/model.sh --hug base --wes 256 --rv 0  
 sh ./online.sh run/40/model.sh --hug base --wes 256 --rv 1  
 sh ./online.sh run/40/model.sh --hug base --wes 400 --rv 0  
 sh ./online.sh run/40/model.sh --hug base --wes 400 --rv 1  

### 集成  
#注意将12个模型放到../working/online/40/下   
#模型集成结果将会放到 ../working/online/40/ensemble/result.zip  
sh ensemble-onine.sh 40  
 
