folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=bread.DeeplabV3Plus \
  --backbone=resnet50 \
  --backbone_weights=places365 \
  --mname=$x \
  $*
