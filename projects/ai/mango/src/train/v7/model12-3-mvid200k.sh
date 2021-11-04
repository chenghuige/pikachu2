folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --emb_size=128 \
  --hash_vid=0 \
  --max_vid=200000 \
  --model=Model12_3 \
  --pooling=dot \
  --mname=$x \
  $*
