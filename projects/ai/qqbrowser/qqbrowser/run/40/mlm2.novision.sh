folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --use_relevance2 \
  --adjust_label=0 \
  --weight_loss=0 \
  --merge_vision \
  --incl=words,vision,merge_vision,merge \
  --merge_method=3 \
  --continue_version=0.novision \
  --decay_epochs=5 \
  --rv=0 \
  --mname=$x \
  $*

