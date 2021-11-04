folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --use_timespan_emb=1 \
    --use_task_mlp=1 \
    --field_concat=1 \
    --use_time_emb=1 \
    --time_bins_per_day=4 \
    --time_bin_shift_hours=1 \
    --time_smoothing=0 \
    --use_slim_fm=1 \
    --model_name=$x \
    $*
    
