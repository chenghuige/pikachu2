folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --use_first_vision \
  --mlp_dims=512 \
  --mname=$x \
  $*

