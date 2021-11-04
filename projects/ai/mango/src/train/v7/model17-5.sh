folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --emb_size=128 \
  --max_vid=100000 \
  --model=Model17_5 \
  --pooling=dot \
  --mname=$x \
  $*
