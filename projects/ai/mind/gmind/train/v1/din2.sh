folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=Model \
  --pooling=dot \
  --use_history \
  --his_pooling=din \
  --din_normalize \
  --batch_size=1024 \
  --mname=$x \
  $*
