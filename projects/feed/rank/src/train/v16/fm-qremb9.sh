folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --feature_dict_size=20000000 \
    --num_feature_buckets=3000000 \
    --model_name=fm.qremb9 \
    $*
    
