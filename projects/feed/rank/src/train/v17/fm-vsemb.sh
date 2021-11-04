folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --vars_split_strategy=embeddings \
    --model_name=fm.vsemb \
    $*
    
