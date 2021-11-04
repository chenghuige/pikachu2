folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

model=WideDeep

sh ./train/${v}/common.sh \
    --use_onehot_emb=1 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --use_timespan_emb=1 \
    --use_time_emb=1 \
    --time_bins_per_day=4 \
    --time_bin_shift_hours=1 \
    --time_smoothing=0 \
    --model=$model \
    --duration_weight=0 \
    --use_wide_position_emb=0 \
    --use_deep_position_emb=0 \
    --use_wd_position_emb=0 \
    --position_combiner=concat \
    --use_wide_val=1 \
    --use_deep_val=0 \
    --ignore_zero_value_feat=0 \
    --deep_out_dim=1 \
    --model_name=$x \
    $*
    
