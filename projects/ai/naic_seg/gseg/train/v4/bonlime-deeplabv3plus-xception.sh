folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=bonlime.DeeplabV3Plus \
  --backbone=xception \
  --normalize_image=-1-1 \
  --batch_size=16 \
  --mname=$x \
  $*
