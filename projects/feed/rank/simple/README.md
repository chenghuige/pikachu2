# genearte record  
sh ./prepare/gen-records.sh  
# train  
sh run.sh  
# evaluate only 
METRIC=1 horovodrun -np 8  sh ./train/v4/sparse-lazy.sh
