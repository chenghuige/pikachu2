folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --emb_size=128 \
  --model=Model16_1 \
  --pooling=dot \
  --mname=$x \
  $*
