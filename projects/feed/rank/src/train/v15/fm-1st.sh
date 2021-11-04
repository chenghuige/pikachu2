folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm-val.sh \
    --use_fm_first_order=1 \
    --model_name=fm.1st \
    $*
    
