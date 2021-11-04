folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --learning_rate=0.001 \
  --min_count=5 \
  --use_w2v \
  --words_w2v \
  --pooling=dot \
  --model=Model \
  --mname=$x \
  $*
