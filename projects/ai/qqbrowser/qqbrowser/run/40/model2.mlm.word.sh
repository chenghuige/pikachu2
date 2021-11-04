folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --words_bert=base \
  --mlmloss_rate=0.1 \
  --model=model2 \
  --relevance=relevance2 \
  --weight_loss \
  --merge_vision \
  --ft_epochs=5 \
  --rv=0 \
  --mname=$x \
  $*

