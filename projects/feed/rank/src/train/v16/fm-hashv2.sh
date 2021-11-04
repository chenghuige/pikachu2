folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --hash_embedding_v2=1 \
    --model_name=fm.hashv2 \
    $*
    
