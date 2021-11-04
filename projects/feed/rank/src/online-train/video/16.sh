#!/bin/sh
source /root/.bashrc
chgenv
export LANG=zh_CN.UTF-8
source ./config.sh

start_hour=$1

export DIR=$base_dir_video

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

abtestid=16

model_dir=$DIR/model/${abtestid}
version=video.ab${abtestid}.taskmlp-fcat-timespan

#   --masked_fields=4621,1145,8466,1761,4391,2551,5024,5468,5001,2818,4729,1589,2957,2834,3259,6148,2815,2981,3819,3763,8873,8273,8802,3778 \
#version=video.wd.youtube.power3.mp01.multi05.mcd20
# version=video.wd.youtube.power3.mp1
# version=video.wd.multi.mdr0.power3.mp01
# version=video.wd.youtube.nolog

# --duration_weight_nolog=1 \
# --duration_weight_multiplier=1 \
# --duration_weight_multiplier=0.1 \

# --duration_weight_nolog=0 \
# --duration_weight_power=3 \
# --duration_weight_multiplier=1 \

# --duration_weight_multiplier=0.1 \
# --multi_obj_duration=1 \
# --multi_obj_duration_loss=sigmoid_cross_entropy \
# --multi_obj_duration_ratio=0.3 \

# remove MITATTP MITATPORNSTP
## The more duration_weight_multiplier the more prefer to duration TODO try 0.1
#  --masked_fields=37,55,61 \  
sh ./train/widedeep/dense-weight-hash.sh \
  --valid_hour=${start_hour} \
  --is_video=1 \
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
  --duration_weight_obj_nolog=1 \
  --duration_scale=minmax \
  --multi_obj_duration2=1 \
  --multi_obj_merge_method=mul \
  --duration_weight=0 \
  --multi_obj_duration_infer_ratio=0.5 \
  --multi_obj_jump_infer_power=0.65 \
  --use_user_emb=1 \
  --use_doc_emb=1 \
  --use_history_emb=1 \
  --use_time_emb=0 \
  --use_timespan_emb=1 \
  --use_task_mlp=1 \
  --field_concat=1 \
  --multi_obj_duration_ratio=0.1 \
  --click_power=1.3 \
  --dur_power=0.7 \
  --cb_click_power=1.5 \
  --cb_dur_power=0.5 \
  --hidden_size=32 \
  --use_wide_position_emb=0 \
  --use_deep_position_emb=0 \
  --use_wd_position_emb=0 \
  --position_combiner=concat \
  --compat_old_model=0 \
  --version=$version \
  --use_step_file=1 \
  --change_cb_user_weight=1 \
  --cb_user_weight=0.1 \
  --l2_reg=0 \
  --masked_fields=3778 \
  $*

echo $version
