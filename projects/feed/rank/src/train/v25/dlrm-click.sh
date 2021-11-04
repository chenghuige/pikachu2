folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --multi_obj_duration_ratio=0. \
    --click_power=1. \
    --dur_power=0. \
    --model_name=$x \
    $*
    
