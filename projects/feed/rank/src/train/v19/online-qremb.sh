folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/online.sh \
    --use_qr_embedding=1 \
    --model_name=$x \
    $*
    
