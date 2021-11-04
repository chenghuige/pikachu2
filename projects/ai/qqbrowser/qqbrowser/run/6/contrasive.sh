folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --label_strategy=all_tags \
  --num_negs=100 \
  --loss_scale=100 \
  --contrasive_rate=0.5 \
  --remove_pred \
  --use_pyfunc=0 \
  --mname=$x \
  $*

