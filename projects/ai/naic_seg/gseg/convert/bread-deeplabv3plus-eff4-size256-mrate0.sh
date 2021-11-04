folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./convert/common.sh \
  --model=bread.DeeplabV3Plus \
  --backbone=EfficientNetB4 \
  --mrate=0 \
  --mname=$x \
  --pretrain=$1 \
  $*
