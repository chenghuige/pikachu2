folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --field_concat=1 \
    --use_task_mlp=1 \
    --use_slim_fm=1 \
    --fm_before_mlp=0 \
    --use_fm_first_order=0 \
    --model_name=torch-$x \
    $*
    
