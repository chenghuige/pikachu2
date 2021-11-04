folder=$(dirname "$0")
v=${folder##*/}

TORCH=1 sh ./train/${v}/train.sh \
    --keras_linear=1 \
    --sparse_emb=1 \
    --optimizers=bert-Adam,bert-SparseAdam \
    --learning_rates=0.001,0.01 \
    --async_valid=0 \
    --model_name=torch \
    $*
    
