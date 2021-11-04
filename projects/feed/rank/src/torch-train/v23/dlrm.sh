folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/fm-multi.sh \
    --fields_pooling=dot \
    --mlp_dims=512,256,64 \
    --task_mlp_dims=16 \
    --model_name=torch-$x \
    $*
    
