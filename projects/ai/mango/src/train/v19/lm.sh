folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --async_valid=0 \
  --metric_eval=0 \
  --train_loop=0 \
  --lm_target=watch_vids \
  --big_model=1 \
  --his_encoder=lstm \
  --allow_cpu=0 \
  --model=Model \
  --mname=$x \
  $*
