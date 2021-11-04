folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
x=${x%.*}

sh ./train/${v}/train.sh \
    --click_power=1. \
    --dur_power=0.6 \
    --compat_old_model=1 \
    --ignore_zero_value_feat=1 \
    --wide_addval=1 \
    --deep_addval=0 \
    --model_name=$x \
    $*
    
