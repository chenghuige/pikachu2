folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/old.sh \
    --click_power=1.1 \
    --model_name=old2 \
    $*
    
