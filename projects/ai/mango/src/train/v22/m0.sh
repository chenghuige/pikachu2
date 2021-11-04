folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/base.sh \
  --use_uid \
  --use_vid \
  --model=Model \
  --mname=$x \
  $*
