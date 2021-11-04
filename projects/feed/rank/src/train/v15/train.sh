folder=$(dirname "$0")
v=${folder##*/}

mark="${MARK:-tuwen}"
#mark="${MARK}"

source ./conf/config.sh ${mark}
train_dir=$DIR/tfrecords

model=WideDeep
bin=train.py
if [ $TORCH ];then
  echo 'torch mode' $TORCH
  if [[ $TORCH == "1" ]]
  then
    bin=torch-train.py
  fi
fi

if [[ $mark == "tuwen" ]]
then
  click_power="1.3"
  dur_power="0.7"
else
  click_power="1.3"
  dur_power="0.7"
fi

echo $bin

python $bin \
    --valid_hour=$valid_hour \
    --restore_exclude=global_step,ignore,learning_rate \
    --hash_encoding=1 \
    --feature_dict_size=20000000 \
    --num_feature_buckets=3000000 \
    --field_dict_size=10000 \
    --duration_weight=1 \
    --sparse_to_dense=1 \
    --deep_final_act=0 \
    --mlp_dims=50 \
    --mlp_drop=0. \
    --field_emb=0 \
    --pooling=sum \
    --dense_activation=relu \
    --change_cb_user_weight=1 \
    --cb_user_weight=0.1 \
    --l2_reg=0 \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --opt_weight_decay=0. \
    --opt_epsilon=1e-6 \
    --min_learning_rate=1e-6 \
    --warmup_proportion=0.1 \
    --learning_rate=0.001 \
    --learning_rate_method=none \
    --dynamic_pad=1 \
    --valid_multiplier=1 \
    --model=$model \
    --num_epochs=1 \
    --interval_steps=1000000 \
    --valid_interval_steps=1000000 \
    --valid_interval_epochs=1 \
    --first_interval_epoch=-1 \
    --train_input=$train_dir, \
    --valid_input=$valid_dir, \
    --batch_size=512 \
    --disable_model_suffix=1 \
    --freeze_graph_final=0 \
    --eval_group=1 \
    --async_valid=1 \
    --write_valid=1 \
    --sync_hdfs=1 \
    --model_dir=$DIR/exps/${v}/base \
    --ori_log_dir=$cloud_dir/exps/${v} \
    --ori_model_dir=$cloud_dir/exps/${v} \
    --valid_span=2 \
    --del_inter_events=1 \
    --write_metric_summary=0 \
    --write_valid=1 \
    --model_name=base \
    --valid_every_hash_n=0 \
    --version=$version \
    --train_loop=1 \
    --loop_type=hour \
    --no_online_result=0 \
    --base_result_dir=$base_result_dir \
    --ev_first=0 \
    --del_inter_model=1 \
    --save_interval_epochs=-1 \
    --save_interval_steps=100000000000 \
    --min_filter_duration=5 \
    --min_click_duration=1 \
    --interests_weight=0 \
    --interests_weight_type=drop \
    --min_interests=0 \
    --min_interests_weight=0.1 \
    --duration_weight_nolog=0 \
    --duration_weight_power=1 \
    --duration_weight_multiplier=1 \
    --dur_loss_no_dur_weight=1 \
    --click_loss_no_dur_weight=1 \
    --duration_log_max=8 \
    --multi_obj_duration_loss=jump \
    --duration_weight_obj_nolog=1 \
    --duration_scale=minmax \
    --multi_obj_duration=0 \
    --multi_obj_duration2=1 \
    --multi_obj_merge_method=mul \
    --duration_weight=0 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --multi_obj_duration_ratio=0.5 \
    --click_power=${click_power} \
    --dur_power=${dur_power} \
    --hidden_size=32 \
    --use_wide_position_emb=0 \
    --use_deep_position_emb=0 \
    --use_wd_position_emb=0 \
    --position_combiner=concat \
    --wide_addval=0 \
    --deep_addval=0 \
    --use_all_data=1 \
    --start_hour=2019122418 \
    --rounds=12 \
    --compat_old_model=0 \
   $*
echo $ts_hour
