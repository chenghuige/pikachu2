folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd-multi.sh \
    --wide_addval=1 \
    --deep_addval=1 \
    --hash_embedding_type=QREmbedding \
    --use_onehot_emb=1 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --use_timespan_emb=1 \
    --use_time_emb=1 \
    --time_bins_per_day=4 \
    --time_bin_shift_hours=1 \
    --time_smoothing=0 \
    --use_task_mlp=1 \
    --field_concat=1 \
    --fm_before_mlp=0 \
    --use_slim_fm=1 \
    --model_name=$x \
    $*
    
