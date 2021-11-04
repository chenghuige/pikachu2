folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/common.sh \
  --use_bert_lr \
  --remove_pred \
  --mname=$x \
  $*


