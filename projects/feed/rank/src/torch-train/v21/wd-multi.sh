folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./train/${v}/wd-multi.sh \
    --multi_obj_duration_loss=BCEWithLogitsLoss \
    --dense_activation=LeakyReLU \
    --save_interval_epochs=1 \
    --keras_linear=1 \
    --sparse_emb=0 \
    --hash_embedding_type=SimpleEmbedding \
    --optimizers=bert-SGD,bert-SGD \
    --opt_momentum=0.9 \
    --learning_rates=0.1,0.1 \
    --use_user_emb=0 \
    --use_doc_emb=0 \
    --use_history_emb=0 \
    --fm_before_mlp=1 \
    --model_name=torch-$x \
    $*
    
