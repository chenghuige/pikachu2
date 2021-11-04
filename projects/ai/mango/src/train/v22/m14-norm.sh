folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_history \
  --use_vid \
  --use_uinfo \
  --use_prev_info \
  --use_class_info \
  --use_stars \
  --use_title \
  --use_story \
  --use_others \
  --use_crosses=0 \
  --use_image=0 \
  --use_history_class \
  --use_last_class \
  --use_last_vid \
  --use_dense \
  --use_dense_his_interval \
  --use_dense_his_durs \
  --use_dense_prev_info \
  --his_encoder='' \
  --his_pooling=din \
  --l2_normalize_before_pooling \
  --model=Model \
  --mname=$x \
  $*
