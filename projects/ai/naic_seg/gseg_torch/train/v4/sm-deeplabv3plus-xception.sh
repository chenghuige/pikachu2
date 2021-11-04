folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=DeepLabV3Plus \
  --backbone=xception \
  --mname=$x \
  $*
