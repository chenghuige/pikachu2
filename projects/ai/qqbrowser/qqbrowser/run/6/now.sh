folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --label_strategy=all_tags \
  --num_negs=1000 \
  --loss_scale=1000 \
  --epochs=4 \
  --decay_epochs=8 \
  --mname=$x \
  $*

