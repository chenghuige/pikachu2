folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_vid \
  --use_history \
  --min_count=5 \
  --use_unk=1 \
  --his_pooling='' \
  --model=Model \
  --mname=$x \
  $*
