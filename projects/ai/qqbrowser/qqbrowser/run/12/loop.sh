folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --cf \
  --bs=8 \
  --epochs=1000000000 \
  --wandb=0 \
  --write_summary=0 \
  --write_metrics_summary=0 \
  --model=model \
  --label_strategy=all_tags \
  --mname=$x \
  $*

