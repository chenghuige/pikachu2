folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --weight_loss \
  --adjust_label \
  --mname=$x \
  $*

