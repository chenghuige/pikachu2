folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm-val.sh \
    --ignore_zero_value_feat=1 \
    --model_name=fm.val.no0 \
    $*
    
