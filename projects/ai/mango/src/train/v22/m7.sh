folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_vid \
  --use_history \
  --use_w2v=0 \
  --his_pooling='' \
  --model=Model \
  --mname=$x \
  $*
