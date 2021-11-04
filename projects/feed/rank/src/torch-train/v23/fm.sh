folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --onehot_fields_pooling=1 \
    --fields_pooling=concat \
    --fields_pooling_after_mlp=fm \
    --use_task_mlp=1 \
    --use_fm_first_order=0 \
    --model_name=torch-$x \
    $*
    
