folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --parse_strategy=3 \
  --epochs=1 \
  --decay_epochs=1 \
  --lr_mul=0.1 \
  --pretrain=now \
  --ev_first \
  --loss_scale=1 \
  --mname=$x \
  $*

