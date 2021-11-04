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
  --use_asr \
  --asr_len=64 \
  --mname=$x \
  $*

