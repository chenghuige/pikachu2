folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --onehot_fields_pooling=1 \
    --fields_pooling=concat \
    --fields_pooling_after_mlp=fm \
    --use_task_mlp=1 \
    --use_fm_first_order=0 \
    --model_name=$x \
    $*
    
