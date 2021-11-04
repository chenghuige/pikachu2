folder=$(dirname "$0")
v=${folder##*/}

# 2 ** 31 = 2147483648
# 2 ** 32 = 4294967296
sh ./train/${v}/fm.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --feature_dict_size=400000000 \
    --num_feature_buckets=20000 \
    --model_name=fm.qremb8 \
    $*
    
