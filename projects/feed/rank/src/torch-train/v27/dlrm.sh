folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./train/${v}/$x.sh \
    --hpooling=sum \
    --model_name=torch-$x \
    --multi_obj_duration_loss=BCEWithLogitsLoss \
    --dense_activation=LeakyReLU \
    --save_interval_epochs=1 \
    --keras_linear=1 \
    --sparse_emb=1 \
    --num_optimizers=2 \
    --optimizers=bert-Adam,bert-SparseAdam \
    --opt_momentum=0. \
    --learning_rates=0.001,0.01 \
    --vars_split_strategy=emb \
    $*
    
