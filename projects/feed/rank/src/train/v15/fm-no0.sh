folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --ignore_zero_value_feat=1 \
    --model_name=fm.no0 \
    $*
    
