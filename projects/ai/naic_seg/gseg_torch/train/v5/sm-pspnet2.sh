folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=sm.PSPNet \
  --backbone=resnet50 \
  --image_size=288,288 \
  --mname=$x \
  $*
