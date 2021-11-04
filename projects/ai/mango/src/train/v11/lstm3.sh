folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --learning_rate=0.001 \
  --history_encoder=lstm \
  --model=Model2 \
  --mname=$x \
  $*
