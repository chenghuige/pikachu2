folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_history \
  --use_vid \
  --use_uinfo \
  --use_class_info \
  --use_stars \
  --use_title \
  --use_story \
  --use_others \
  --use_active=0 \
  --his_pooling='' \
  --model=Model \
  --mname=$x \
  $*
