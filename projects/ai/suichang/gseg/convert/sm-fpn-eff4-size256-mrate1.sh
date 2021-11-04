folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./convert/common.sh \
  --model=sm.FPN \
  --backbone=EfficientnetB4 \
  --mrate=1 \
  --mname=$x \
  --pretrain=$1 \
  $*
