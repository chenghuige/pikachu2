folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/train.sh \
    --model_name=base \
    $*
    
