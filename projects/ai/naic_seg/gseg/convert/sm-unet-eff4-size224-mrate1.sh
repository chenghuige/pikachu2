folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./convert/common.sh \
  --model=sm.Unet \
  --backbone=EfficientnetB4 \
  --image_size=224,224 \
  --mrate=1 \
  --mname=$x \
  --pretrain=$1 \
  $*
