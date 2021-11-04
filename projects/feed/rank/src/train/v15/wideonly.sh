folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --wide_only=1 \
    --model_name=fm.wideonly \
    $*
    
