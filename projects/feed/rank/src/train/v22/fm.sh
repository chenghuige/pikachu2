folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd.sh \
    --field_concat=1 \
    --use_slim_fm=1 \
    --use_fm_first_order=0 \
    --fm_before_mlp=0 \
    --model_name=$x \
    $*
    
