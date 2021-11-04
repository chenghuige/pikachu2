folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --hash_vid=0 \
  --model=Model6_8 \
  --pooling=dot \
  --mname=$x \
  $*
