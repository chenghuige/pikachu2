#!/bin/sh
source /root/.bashrc
chgenv
export LANG=zh_CN.UTF-8
source ./config.sh

start_hour=$1

export DIR=$base_dir_video

if (($METRIC==1))
then
  train_dir=$DIR/${start_hour}/tfrecords
  valid_dir=$train_dir
  num_valid=0
else
 train_dir=$DIR/${start_hour}/tfrecords
 valid_dir=$train_dir
 num_valid=500000
fi

model_dir=$DIR/model
version=video.ab16.mlp005.power3
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
  --interests_weight=1 \
  --interests_weight_type=clip \
  --min_interests=0 \
  --duration_weight_nolog=0 \
  --duration_weight_power=3 \
  --duration_weight_multiplier=0.05 \
  --multi_obj_duration=1 \
  --multi_obj_duration_loss=sigmoid_cross_entropy \
  --multi_obj_duration_ratio=1. \
  --masked_fields=3778 \
  --use_user_emb=1 \
  --use_doc_emb=1 \
  --use_history_emb=1 \
  --hidden_size=32 \
  --use_wide_position_emb=0 \
  --use_deep_position_emb=0 \
  --use_wd_position_emb=0 \
  --position_combiner=concat \
  --version=$version \
  --use_step_file=1 \
  $*

echo $version
