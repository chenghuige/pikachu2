folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --sparse_emb=1 \
    --optimizers=bert-SGD,bert-SGD \
    --opt_momentum=0. \
    --learning_rates=0.1,0.1 \
    --num_optimizers=1 \
    --deep_only=1 \
    --use_onehot_emb=0 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=0 \
    --model_name=torch-$x \
    $*
    
