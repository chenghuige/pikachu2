# baseline 0.7518  
horovodrun -np 8 sh ./train/base.sh
# add doc emb 0.7519  
horovodrun -np 8 sh ./train/doc-emb.sh
