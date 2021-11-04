folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --fields_pooling_after_mlp='' \
    --mlp_dims=64 \
    --task_mlp_dims=16 \
    --model_name=$x \
    $*
    
