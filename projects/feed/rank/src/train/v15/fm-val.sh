folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --wide_addval=1 \
    --deep_addval=1 \
    --model_name=fm.val \
    $*
    
