folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --hidden_size=768 \
  --layer_norm2=0 \
  --mname=$x \
  $*

