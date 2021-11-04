folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=model \
  --use_bert_lr \
  --epochs=3 \
  --num_epochs=8 \
  --mname=$x \
  $*

