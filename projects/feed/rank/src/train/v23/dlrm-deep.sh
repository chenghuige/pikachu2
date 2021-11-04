folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --deep_only \
    --fields_pooling_after_mlp='' \
    --model_name=$x \
    $*
    
