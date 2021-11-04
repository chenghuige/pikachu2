folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --hidden_size=768 \
  --mlp_dims2=512 \
  --layer_norm2 \
  --mname=$x \
  $*

