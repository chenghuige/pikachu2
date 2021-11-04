folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo "torch: ${x}"
x=${x%.*}

TORCH=1 sh ./torch-train/${v}/wd.sh \
    --hash_embedding_type=QREmbedding \
    --wide_addval=1 \
    --deep_addval=0 \
    --sparse_emb=1 \
    --num_optimizers=2 \
    --optimizers=bert-Adam,bert-SparseAdam \
    --opt_momentum=0. \
    --learning_rates=0.001,0.01 \
    --vars_split_strategy=emb \
    --use_onehot_emb=1 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --use_timespan_emb=1 \
    --use_task_mlp=1 \
    --field_concat=1 \
    --use_time_emb=1 \
    --time_bins_per_day=4 \
    --time_bin_shift_hours=1 \
    --time_smoothing=0 \
    --use_slim_fm=1 \
    --fm_before_mlp=0 \
    --model_name=torch-$x \
    $*
    
