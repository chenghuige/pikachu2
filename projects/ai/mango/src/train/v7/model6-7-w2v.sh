folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --use_w2v=1 \
  --hash_vid=0 \
  --model=Model6_7 \
  --pooling=dot \
  --mname=$x \
  $*
