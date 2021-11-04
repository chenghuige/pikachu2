folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --change_cb_user_weight=0 \
    --use_deep_val=1 \
    --fields_pooling_after_mlp='' \
    --model_name=$x \
    $*
    
