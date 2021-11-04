folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --optimizers=bert-adagrad,bert-adagrad \
    --model_name=fm.adagrad \
    $*
    
