folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/udonly.sh \
    --hash_embedding_v2_ud=1 \
    --model_name=udonly.hashv2 \
    $*
    
