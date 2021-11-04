folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --use_relevance2 \
  --adjust_label=0 \
  --weight_loss=1 \
  --merge_vision \
  --incl=merge \
  --merge_method=3 \
  --ft_epochs=5 \
  --rv=0 \
  --mname=$x \
  $*

