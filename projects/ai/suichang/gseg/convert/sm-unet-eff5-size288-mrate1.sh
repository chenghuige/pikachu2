folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./convert/common.sh \
  --model=sm.Unet \
  --backbone=EfficientnetB5 \
  --image_size=288,288 \
  --mrate=1 \
  --mname=$x \
  --pretrain=$1 \
  $*
