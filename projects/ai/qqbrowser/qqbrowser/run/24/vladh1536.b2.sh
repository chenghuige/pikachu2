folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base2.sh \
  --vlad_hidden_size=1536 \
  --mname=$x \
  $*

