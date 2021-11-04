folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --loss_fn=multi \
  --loss_tags=0 \
  --num_negs=100 \
  --mname=$x \
  $*

