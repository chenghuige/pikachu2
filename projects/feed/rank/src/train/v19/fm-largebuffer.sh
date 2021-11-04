folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/fm.sh \
    --buffer_size=3000000 \
    --model_name=$x \
    $*
    
