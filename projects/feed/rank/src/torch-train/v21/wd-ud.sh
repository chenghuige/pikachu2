folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --model_name=torch-$x \
    $*
    
