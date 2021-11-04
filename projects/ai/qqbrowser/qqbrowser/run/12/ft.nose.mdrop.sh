folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/ft.base.sh \
  --pretrain=now.nose.mdrop \
  --mdrop_rate=0.1 \
  --use_se=0 \
  --from_logits=0 \
  --loss_fn=corr \
  --mname=$x \
  $*

