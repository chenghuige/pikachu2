folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --wide_addval=1 \
    --deep_addval=1 \
    --model_name=$x \
    $*
    
