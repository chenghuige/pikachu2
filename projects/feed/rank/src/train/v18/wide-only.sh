folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
x=${x%.*}

sh ./train/${v}/wide-deep.sh \
    --wide_only=1 \
    --model_name=$x \
    $*
    
