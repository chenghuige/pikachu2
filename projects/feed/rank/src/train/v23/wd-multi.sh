folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

if [[ $mark == "tuwen" ]]
then
  click_power="1.3"
  dur_power="0.7"
else
  click_power="1.3"
  dur_power="0.7"
fi

sh ./train/${v}/wd.sh \
    --use_task_mlp=1 \
    --duration_log_max=8 \
    --multi_obj_type=shared_bottom \
    --use_jump_loss=1 \
    --multi_obj_duration_loss=sigmoid_cross_entropy \
    --duration_weight_obj_nolog=1 \
    --duration_scale=minmax \
    --multi_obj_merge_method=mul \
    --duration_weight=0 \
    --multi_obj_duration_ratio=0.5 \
    --click_power=${click_power} \
    --dur_power=${dur_power} \
    --model_name=$x \
    $*
    
