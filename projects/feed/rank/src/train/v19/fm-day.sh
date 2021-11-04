folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/fm.sh \
    --loop_type=day \
    --model_name=$x \
    $*
    
