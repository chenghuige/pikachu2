folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --use_cold_emb=1 \
    --use_deep_val=1 \
    --fields_pooling_after_mlp='' \
    --model_name=$x \
    $*
    
