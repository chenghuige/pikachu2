folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --decay_epochs=1 \
  --use_se=0 \
  --label_strategy=all_tags \
  --mname=$x \
  $*

