folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --parse_trategy=3 \
  --epochs=1 \
  --decay_epochs=1 \
  --lr_mul=0.1 \
  --pretrain=now \
  --ev_first \
  --label_strategy=all_tags \
  --num_negs=1000 \
  --loss_scale=1 \
  --mname=$x \
  $*

