folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=sm.Unet \
  --backbone=resnet50 \
  --kernel_size=3 \
  --mname=$x \
  $*
