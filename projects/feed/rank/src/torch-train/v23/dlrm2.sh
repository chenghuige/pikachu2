folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/dlrm.sh \
    --fields_pooling_after_mlp='' \
    --model_name=torch-$x \
    $*
    
