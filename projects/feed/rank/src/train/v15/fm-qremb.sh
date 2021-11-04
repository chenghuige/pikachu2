folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --model_name=fm.qremb \
    $*
    
