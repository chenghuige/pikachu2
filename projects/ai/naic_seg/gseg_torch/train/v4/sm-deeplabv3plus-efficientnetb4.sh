folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=DeepLabV3Plus \
  --backbone=efficientnet-b4 \
  --batch_size=16 \
  --mname=$x \
  $*
