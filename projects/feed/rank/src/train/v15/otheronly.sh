folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --use_onehot_emb=0 \
    --deep_only=1 \
    --model_name=otheronly \
    $*
    
