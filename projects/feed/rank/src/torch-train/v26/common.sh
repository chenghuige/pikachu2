folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./train/${v}/common.sh \
    --save_interval_epochs=1 \
    --keras_linear=1 \
    --sparse_emb=0 \
    --optimizers=bert-SGD,bert-SGD \
    --opt_momentum=0.9 \
    --learning_rates=0.1,0.1 \
    --model_name=torch-$x \
    $*
    
