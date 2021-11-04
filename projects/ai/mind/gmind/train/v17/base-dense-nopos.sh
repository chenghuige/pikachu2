folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --model=Model \
  --use_dense \
  --dense_use_his_len \
  --dense_use_impression \
  --max_lookup_history=100 \
  --mname=$x \
  $*
