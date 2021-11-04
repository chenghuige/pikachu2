folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --sparse_emb=1 \
    --num_optimizers=2 \
    --optimizers=bert-SGD,bert-SparseAdam \
    --opt_momentum=0.9 \
    --learning_rates=0.001,0.1 \
    --deep_only=1 \
    --use_onehot_emb=0 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=0 \
    --model_name=torch-$x \
    $*
    
