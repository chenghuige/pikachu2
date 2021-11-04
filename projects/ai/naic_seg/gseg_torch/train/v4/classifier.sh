folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=classifier \
  --backbone=ResNet50 \
  --vie=1 \
  --mname=$x \
  $*
