folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --onehot_fields_pooling=1 \
    --fields_pooling=dot \
    --fields_pooling_after_mlp='' \
    --mlp_dims=512,256 \
    --task_mlp_dims=64,16 \
    --masked_fields='last' \
    --mask_mode=regex-excl \
    --use_wide_val=1 \
    --use_deep_val=1 \
    --use_cold_emb=1 \
    --use_product_emb=1 \
    --change_cb_user_weight=0 \
    --model_name=$x \
    $*
    
