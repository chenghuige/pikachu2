folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --use_qr_embedding=0 \
    --click_power=1. \
    --dur_power=0.6 \
    --compat_old_model=1 \
    --ignore_zero_value_feat=1 \
    --wide_addval=1 \
    --deep_addval=0 \
    --model_name=$x \
    $*
    
