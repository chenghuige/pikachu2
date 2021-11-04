folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./train/${v}/wd.sh \
    --save_interval_epochs=1 \
    --keras_linear=1 \
    --sparse_emb=0 \
    --optimizers=bert-Adam,bert-Adam \
    --opt_momentum=0. \
    --learning_rates=0.001,0.01 \
    --use_user_emb=0 \
    --use_doc_emb=0 \
    --use_history_emb=0 \
    --model_name=torch-$x \
    $*
    
