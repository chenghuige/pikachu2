folder=$(dirname "$0")
v=${folder##*/}

x=$(basename "$0")
echo $x

mark="${MARK:-video}"

source ./conf/config.sh ${mark}
train_dir=$DIR/tfrecords

bin=./train.py
if [ $TORCH ];then
  if [[ $TORCH == "1" ]]
  then
     bin=./torch-train.py
  fi
fi

echo $bin

$bin \
    --eval_group_by_impression \
    --eval_online \
    --base_result_dir=${base_result_dir} \
    --base_data_name=${product} \
    --other_data_names=${product_others} \
    --start_hour=${start_hour} \
    --rounds=12 \
    --debug=0 \
    --verbose=0 \
    --parallel_read_files=1 \
    --eval_days=1 \
    --write_summary=1 \
    --write_metric_summary=1 \
    --write_online_summary=0 \
    --metric_eval=1 \
    --monitor_l2=1 \
    --monitor_gradients=0 \
    --monitor_global_gradients=0 \
    --allow_growth=1 \
    --max_used_gpu_mem=220 \
    --interval_steps=1000 \
    --valid_interval_steps=1000 \
    --restore_exclude=global_step,ignore,learning_rate \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --opt_weight_decay=0. \
    --opt_epsilon=1e-6 \
    --min_learning_rate=1e-6 \
    --warmup_proportion=0.1 \
    --learning_rate=0.001 \
    --learning_rate_method=none \
    --valid_multiplier=1 \
    --num_epochs=1 \
    --valid_interval_epochs=1 \
    --first_interval_epoch=-1 \
    --train_input=$train_dir, \
    --valid_input=$valid_dir, \
    --batch_size=512 \
    --disable_model_suffix=1 \
    --freeze_graph_final=0 \
    --eval_group=1 \
    --async_valid=1 \
    --async_eval=1 \
    --cache_valid=0 \
    --write_valid=1 \
    --write_valid_only=0 \
    --sync_hdfs=0 \
    --model_dir=${exps_dir}/${v}/base \
    --ori_log_dir=$cloud_dir/exps/${v} \
    --ori_model_dir=$cloud_dir/exps/${v} \
    --valid_span=2 \
    --del_inter_events=1 \
    --model_name=base \
    --valid_every_hash_n=0 \
    --version=$version \
    --train_loop=1 \
    --loop_type=hour \
    --ev_first=0 \
    --del_inter_model=1 \
    --save_interval_epochs=-1 \
    --save_interval_steps=100000000000 \
    --hvd_device_dense=/cpu:0 \
    --use_all_data=1 \
    --hash_encoding=1 \
    --duration_weight=1 \
    --sparse_to_dense=1 \
    --deep_final_act=0 \
    --mlp_dims=50 \
    --mlp_drop=0. \
    --field_emb=0 \
    --pooling=sum \
    --dense_activation=leaky_relu \
    --change_cb_user_weight=0 \
    --cb_user_weight=0.1 \
    --l2_reg=0 \
    --min_filter_duration=5 \
    --min_click_duration=1 \
    --interests_weight=0 \
    --interests_weight_type=drop \
    --min_interests=0 \
    --min_interests_weight=0.1 \
    --hidden_size=32 \
    --hash_embedding_type=QREmbedding \
    --hash_combiner=mul \
    --feature_dict_size=20000000 \
    --num_feature_buckets=3000000 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --fields_pooling=concat \
    --compat_old_model=0 \
   $*

