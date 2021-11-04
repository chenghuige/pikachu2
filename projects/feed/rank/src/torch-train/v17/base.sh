folder=$(dirname "$0")
v=${folder##*/}

TORCH=1 sh ./train/${v}/train.sh \
    --rounds=1 \
    --save_interval_epochs=1 \
    --keras_linear=1 \
    --sparse_emb=0 \
    --optimizers=bert-SGD,bert-SGD \
    --opt_momentum=0.9 \
    --learning_rates=0.1,0.1 \
    --model_name=torch.base \
    $*
    
