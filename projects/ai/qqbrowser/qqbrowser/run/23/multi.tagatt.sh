folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --loss_fn=multi \
  --loss_tags=1 \
  --tag_pooling=att \
  --mname=$x \
  $*

