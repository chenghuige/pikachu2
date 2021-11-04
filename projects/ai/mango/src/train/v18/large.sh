folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --big_model=1 \
  --his_encoder=lstm \
  --allow_cpu=0 \
  --model=Model \
  --mname=$x \
  $*
