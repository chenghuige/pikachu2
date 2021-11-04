folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --emb_size=128 \
  --min_count=0 \
  --use_w2v \
  --words_w2v \
  --model=Model17_2 \
  --pooling=dot \
  --mname=$x \
  $*
