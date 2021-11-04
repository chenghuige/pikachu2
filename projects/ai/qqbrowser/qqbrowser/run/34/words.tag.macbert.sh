folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/macbert.sh \
  --use_words \
  --tag_w2v \
  --tag_norm \
  --tag_trainable \
  --mname=$x \
  $*

