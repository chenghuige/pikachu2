folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --allow_cpu=0 \
  --dropout=0.3 \
  --history_encoder=lstm \
  --model=Model \
  --mname=$x \
  $*
