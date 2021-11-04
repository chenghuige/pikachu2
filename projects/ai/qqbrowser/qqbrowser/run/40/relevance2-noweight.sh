folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --use_relevance2 \
  --adjust_label=0 \
  --pretrain=base.roberta.rv1.400/pointwise \
  --weight_loss=0 \
  --mname=$x \
  $*

