folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/fm.sh \
    --shuffle=0 \
    --shuffle_files=0 \
    --shuffle_batch=0 \
    --parallel_read_files=0 \
    --model_name=$x \
    $*
    
