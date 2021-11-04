folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
x=${x%.*}

sh ./train/${v}/train.sh \
    --ignore_zero_value_feat=0 \
    --wide_addval=0 \
    --deep_addval=0 \
    --model_name=$x \
    $*
    
