folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/common.sh \
  --model=sm.FPN \
  --backbone=EfficientNetB0 \
  --batch_size=16 \
  --mname=$x \
  $*
