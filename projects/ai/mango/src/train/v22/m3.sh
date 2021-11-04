folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_vid \
  --use_history \
  --his_pooling='' \
  --use_vocab_emb \
  --min_count=5 \
  --model=Model \
  --mname=$x \
  $*
