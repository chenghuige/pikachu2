folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/dlrm.sh \
    --model_name=torch-$x \
    --large_emb=1 \
    --feature_dict_size=80000000 \
    --num_feature_buckets=12000000 \
    $*
    
