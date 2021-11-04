folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/$v/base.sh \
  --lr_decay_power=0.5 \
  --mname=$x \
  $*

