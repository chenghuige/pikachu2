folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --vars_split_strategy=embeddings \
    --learning_rates=0.01,0.001 \
    --model_name=fm.vsemb2 \
    $*
    
