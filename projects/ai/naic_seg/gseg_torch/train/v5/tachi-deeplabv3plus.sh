folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=tachi.DeepLabV3Plus \
  --backbone=ResNet50 \
  --mname=$x \
  $*
