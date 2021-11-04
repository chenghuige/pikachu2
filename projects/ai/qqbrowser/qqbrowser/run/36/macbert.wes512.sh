folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/macbert.sh \
  --word_emb_size=512 \
  --mname=$x \
  $*

