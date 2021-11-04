folder=$(dirname "$0")
v=${folder##*/}

# 3000000 * 60
sh ./train/${v}/fm.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --feature_dict_size=180000000 \
    --num_feature_buckets=60 \
    --model_name=fm.qremb5 \
    $*
    
