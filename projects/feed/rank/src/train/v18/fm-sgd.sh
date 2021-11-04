folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --optimizers=bert-sgd,bert-sgd \
    --model_name=fm.sgd \
    $*
    
