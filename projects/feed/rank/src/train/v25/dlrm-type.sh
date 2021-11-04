folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --use_all_type=1 \
    --use_type_emb=1 \
    --use_network_emb=1 \
    --use_activity_emb=1 \
    --use_refresh_emb=0 \
    --model_name=$x \
    $*
    
