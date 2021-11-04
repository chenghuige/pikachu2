folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --deep_only=1 \
    --use_onehot_emb=0 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=0 \
    --model_name=torch-$x \
    $*
    
