#!/bin/sh
source /root/.bashrc
#chgenv
#blenv
chgenv
export LANG=zh_CN.UTF-8
source ./config.sh

start_hour=$1

#export DIR=$base_dir_tuwen
#export DIR="/search/odin/publicData/CloudS/baili/rank/data/tuwen_hour_sgsapp_v1"
#tf dataset
#export DIR="/search/odin/publicData/CloudS/baili/new_data/Doc"
#export DIR="/search/odin/publicData/CloudS/baili/new_data/tuwen_hour_sgsapp_v1"
export DIR="/search/odin/publicData/CloudS/baili/new_rank/data/tuwen_hour_sgsapp_v1"
#export DIR="/search/odin/publicData/CloudS/baili/rank/sgsapp/data/tuwen_hour_sgsapp_v1"

if (($METRIC==1))
then
  train_dir=$DIR/tfrecords/${start_hour}
  valid_dir=$train_dir
  num_valid=0
else
 train_dir=$DIR/tfrecords/${start_hour}
 valid_dir=$train_dir
 num_valid=500000
fi

shida_data_dir="/search/odin/publicData/CloudS/baili/new_rank/data_shida/tuwen_hour_sgsapp_v1"
#shida_data_dir="/search/odin/publicData/CloudS/baili/rank/shida/data/tuwen_hour_shida_v1"
shida_train_dir=${shida_data_dir}/tfrecords/${start_hour}
if [ -f ${shida_train_dir}/num_records.txt ] ;then
    train_dir=${train_dir},${shida_train_dir}
fi

newmse_data_dir="/search/odin/publicData/CloudS/baili/new_rank/data_newmse/tuwen_hour_sgsapp_v1"
#newmse_data_dir="/search/odin/publicData/CloudS/baili/rank/newmse/data/tuwen_hour_newmse_v1"
newmse_train_dir=${newmse_data_dir}/tfrecords/${start_hour}
if [ -f ${newmse_train_dir}/num_records.txt ] ;then
    train_dir=${train_dir},${newmse_train_dir}
fi

echo 'train_dir' $train_dir
echo 'valid_dir' $valid_dir

abtestid=16


model_dir="/home/gezi/tmp/rank/data/tuwen_hour_sgsapp_v1/exps/v15/debug.old"
version=tuwen.ab${abtestid}

# version=tuwen.wd.youtube.power3.mp1
# power3 is ok at first but group/auc decreasing might due to online learning or hash conflict TODO

# remove MITATTP MITATPORNSTP
#  --masked_fields=37,55,61 \  

# --duration_weight_nolog=0 \
# --duration_weight_power=3 \
# --duration_weight_multiplier=0.1 \

# --duration_weight_nolog=1 \
# --min_filter_duration=5 \

sh ./train/widedeep/dense-weight-hash.sh \
  --valid_hour=${start_hour} \
  --max_models_keep=1 \
  --min_train=10000 \
  --model_dir=$model_dir \
  --train_input=$train_dir \
  --valid_input=$valid_dir \
  --num_valid=$num_valid \
  --field_emb=0 \
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
  --multi_obj_duration=1 \
  --multi_obj_duration_loss=jump \
  --multi_obj_duration_ratio=0.5 \
  --duration_weight_obj_nolog=1 \
  --duration_scale=minmax \
  --multi_obj_duration2=1 \
  --multi_obj_merge_method=mul \
  --duration_weight=0 \
  --multi_obj_duration_infer_ratio=0.1 \
  --multi_obj_jump_infer_power=0.45 \
  --click_power=1.,1.7,1. \
  --dur_power=0.6,0.3,0.6 \
  --cb_click_power=1.,1.9,1. \
  --cb_dur_power=0.6,0.1,0.6 \
  --use_user_emb=1 \
  --use_doc_emb=1 \
  --use_history_emb=1 \
  --use_time_emb=0 \
  --use_timespan_emb=0 \
  --hidden_size=32 \
  --use_wide_position_emb=0 \
  --use_deep_position_emb=0 \
  --use_wd_position_emb=0 \
  --position_combiner=concat \
  --version=$version \
  --use_step_file=1 \
  --change_cb_user_weight=1 \
  --cb_user_weight=0.1 \
  --l2_reg=0 \
  --train_only=1 \
  --compat_old_model=1 \
  --ignore_zero_value_feat=1 \
  --wide_addval=1 \
  --deep_addval=0 \
  $*

echo $version
