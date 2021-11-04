folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

model=WideDeep

sh ./train/${v}/common.sh \
    --model=$model \
    --duration_weight=0 \
    --use_wide_position_emb=0 \
    --use_deep_position_emb=0 \
    --use_wd_position_emb=0 \
    --position_combiner=concat \
    --wide_addval=0 \
    --deep_addval=0 \
    --model_name=$x \
    $*
    
