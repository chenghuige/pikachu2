folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --vars_split_strategy=embeddings \
    --optimizers=bert-momentum,bert-momentum \
    --learning_rates=0.01,0.1 \
    --model_name=fm.vsemb.momentum \
    $*
    
