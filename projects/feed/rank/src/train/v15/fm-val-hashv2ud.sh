folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm-val.sh \
    --hash_embedding_v2_ud=1 \
    --model_name=fm.val.hashv2ud \
    $*
    
