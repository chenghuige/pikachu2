folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./convert/common.sh \
  --model=sm.FPN \
  --backbone=EfficientnetB6 \
  --mrate=0 \
  --mname=$x \
  --pretrain=$1 \
  $*
