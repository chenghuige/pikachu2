folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/fm.sh \
    --use_all_data=0 \
    --model_name=$x \
    $*
    
