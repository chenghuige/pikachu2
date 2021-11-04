folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --deep_only=1 \
    --field_concat=1 \
    --fm_before_mlp=0 \
    --use_slim_fm=1 \
    --use_fm_first_order=0 \
    --model_name=$x \
    $*
    
