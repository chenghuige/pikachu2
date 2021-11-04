folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd.sh \
    --use_qr_embedding=0 \
    --hash_combiner=sum \
    --hash_combiner_ud=sum \
    --use_user_emb=0 \
    --use_doc_emb=0 \
    --use_history_emb=0 \
    --model_name=$x \
    $*
    
