folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --deep_only \
    --use_deep_val=1 \
    --use_onehot_emb=1 \
    --use_other_embs=0 \
    --fields_pooling_after_mlp='' \
    --model_name=$x \
    $*
    
