folder=$(dirname "$0")
v=${folder##*/}

# 1432 * 3000000
sh ./train/${v}/fm.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --feature_dict_size=4296000000 \
    --num_feature_buckets=1432 \
    --model_name=fm.qremb6 \
    $*
    
