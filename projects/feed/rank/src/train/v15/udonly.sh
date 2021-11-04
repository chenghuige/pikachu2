folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/train.sh \
    --id_feature_only=1 \
    --model_name=udonly \
    $*
    
