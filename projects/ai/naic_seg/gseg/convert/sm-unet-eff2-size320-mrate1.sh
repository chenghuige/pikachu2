folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./convert/common.sh \
  --model=sm.Unet \
  --backbone=EfficientnetB2 \
  --image_size=320,320 \
  --mrate=1 \
  --mname=$x \
  --pretrain=$1 \
  $*
