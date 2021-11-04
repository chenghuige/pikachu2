folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=Model \
  --feat_pooling=concat \
  --use_history \
  --use_uid=0 \
  --mname=$x \
  $*
