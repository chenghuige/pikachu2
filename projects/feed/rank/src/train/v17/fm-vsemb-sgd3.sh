folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --vars_split_strategy=embeddings \
    --optimizers=bert-sgd,bert-sgd \
    --learning_rates=0.0001,0.001 \
    --model_name=fm.vsemb.sgd3 \
    $*
    
