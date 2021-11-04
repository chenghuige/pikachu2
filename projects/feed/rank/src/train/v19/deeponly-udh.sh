folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd.sh \
    --use_qr_embedding=1 \
    --hash_combiner=mul \
    --feature_dict_size=3000000 \
    --num_feature_buckets=3000000 \
    --deep_only=1 \
    --use_onehot_emb=0 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --model_name=$x \
    $*
    
