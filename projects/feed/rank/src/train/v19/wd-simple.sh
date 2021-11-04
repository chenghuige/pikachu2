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
    --use_user_emb=0 \
    --use_doc_emb=0 \
    --use_history_emb=0 \
    --model_name=$x \
    $*
    
