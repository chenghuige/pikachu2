folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/ft.base.sh \
  --from_logits=0 \
  --mname=$x \
  $*

