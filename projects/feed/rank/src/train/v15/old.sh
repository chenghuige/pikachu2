folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/train.sh \
    --compat_old_model=1 \
    --ignore_zero_value_feat=1 \
    --wide_addval=1 \
    --deep_addval=0 \
    --model_name=old \
    $*
    
