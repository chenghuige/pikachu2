folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --start_hour=2019122418 \
    --model_name=fm2 \
    $*
    
