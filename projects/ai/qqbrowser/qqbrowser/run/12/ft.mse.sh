folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/ft.base.sh \
  --loss_fn=mse \
  --mname=$x \
  $*

