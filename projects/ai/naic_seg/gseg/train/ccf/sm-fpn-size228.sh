folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --image_size=128,128 \
  --model=sm.FPN \
  --backbone=resnet18 \
  --mname=$x \
  $*
