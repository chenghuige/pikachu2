folder=$(dirname "$0")
v=${folder##*/}

# 2 ** 31 = 2147483648
# 2 ** 32 = 4294967296
sh ./train/${v}/fm.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --feature_dict_size=3000000 \
    --num_feature_buckets=3000000 \
    --model_name=fm.simemb \
    $*
    
