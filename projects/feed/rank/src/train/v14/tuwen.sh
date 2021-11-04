v=14
source ./config.sh
export DIR=$base_dir_tuwen
train_dir=$DIR/tfrecords

model=WideDeep
bin=train.py
if [ $TORCH ];then
  echo 'torch mode' $TORCH
  if [[ $TORCH == '1' ]]
  then
    bin=torch-train.py
  fi
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
    --dynamic_pad=1 \
    --simple_parse=0 \
    --valid_multiplier=1 \
    --deep_final_act=0 \
    --mlp_dims=50 \
    --mlp_drop=0. \
    --field_emb=0 \
    --pooling=sum \
    --dense_activation=relu \
    --model=$model \
    --num_epochs=1 \
    --valid_interval_epochs=1 \
    --first_interval_epoch=-1 \
    --train_input=$train_dir, \
    --valid_input=$valid_dir, \
    --model_dir=$DIR/exps/base \
    --batch_size=512 \
    --max_feat_len=100 \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --opt_weight_decay=0. \
    --opt_epsilon=1e-6 \
    --min_learning_rate=1e-6 \
    --warmup_proportion=0.1 \
    --learning_rate=0.001 \
    --write_valid=0 \
    --disable_model_suffix=1 \
    --eval_group=1 \
    --use_wide_position_emb=0 \
    --use_deep_position_emb=0 \
    --position_combiner=concat \
    --min_filter_duration=5 \
    --min_click_duration=1 \
    --interests_weight=0 \
    --interests_weight_type=clip \
    --min_interests=0 \
    --duration_weight_nolog=1 \
    --duration_weight_multiplier=0.05 \
    --multi_obj_duration=1 \
    --multi_obj_duration_loss=sigmoid_cross_entropy \
    --multi_obj_duration_ratio=0. \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=1 \
    --hidden_size=32 \
    --change_cb_user_weight=1 \
    --cb_user_weight=0.1 \
    $*
