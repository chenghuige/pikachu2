folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --vars_split_strategy=embeddings \
    --optimizers=bert-sgd,bert-sgd \
    --model_name=fm.vsemb.sgd \
    $*
    
