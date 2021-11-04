folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
x=${x%.*}

sh ./train/${v}/base.sh \
    --multi_obj_duration=0 \
    --multi_obj_duration2=0 \
    --model_name=$x \
    $*
    
