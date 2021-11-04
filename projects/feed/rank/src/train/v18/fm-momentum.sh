folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --optimizers=bert-momentum,bert-momentum \
    --learning_rates=0.01,0.1 \
    --model_name=fm.momentum \
    $*
    
